import os
import sys

path = os.path.abspath(".")
sys.path.insert(0,path + "/src/agent/scripts")
import numpy as np
import matplotlib as plt
from numpy.lib.function_base import angle
from scene import Scene
from agv import AGV
from simulator import Simulator
from gym import Env, spaces
from OpenGL.GL import *
from PIL import Image
import math
from simulator import *
from record_data import *

RESET_ACTION = -1
TARGET_X = -5.0
TARGET_Y = 0.0
TARGET_ANGLE = 270.0  # >0

class RobotEnv(Env):
    def __init__(self):
        self.action_dim = 6
        self.observation_space = 1

        self.simulator = Simulator()
        self.agv = AGV()
        self.scene = Scene()

        self.arriveGoal = False
        self.isCollide = False
        self.angleOutofRange = False

        self.stepCount = 0
        self.stepReward = 0.
        
        self.last_dis = math.sqrt(math.pow(AGV_START_POSITION_X - TARGET_X, 2) + math.pow(AGV_START_POSITION_Y - TARGET_X, 2))

        return
    
    def initRobotEnv(self):
        self.scene.loadRawGripMap("./uncut_map_data.txt")
        self.scene.setTarget(TARGET_X, TARGET_Y)
        self.scene.setTarget(TARGET_X+0.1, TARGET_Y)
        self.scene.setTarget(TARGET_X+0.2, TARGET_Y)
        self.scene.setTarget(TARGET_X-0.2, TARGET_Y)
        self.scene.setTarget(TARGET_X-0.1, TARGET_Y)
        self.scene.setTarget(TARGET_X, TARGET_Y+0.1)
        self.scene.setTarget(TARGET_X, TARGET_Y+0.2)
        self.scene.setTarget(TARGET_X, TARGET_Y-0.1)
        self.scene.setTarget(TARGET_X, TARGET_Y-0.2)
        if self.simulator.initialize(self.agv, self.scene) == False:
            print("Error exists when initializing simulator!")
            return False
        return True 

    def step(self, action, pose_x, pose_y, angle):
        self.stepCount += 1
        obs = self.simulator.simulate(self.agv, self.scene, action, pose_x, pose_y, angle)
        self.agv.lastAction = action
        done = self.end()
        reward = self.calculateReward()
        # print("agv position: ", self.agv.x, self.agv.y)

        return obs, reward, done
    
    def calculateReward(self):
        # self.stepReward -= 10
        total_dis = math.sqrt(math.pow(AGV_START_POSITION_X - TARGET_X, 2) + math.pow(AGV_START_POSITION_Y - TARGET_Y, 2))
        # The distance of the agv to the goal
        distance = math.sqrt(math.pow(self.agv.x - TARGET_X, 2) + math.pow(self.agv.y - TARGET_Y, 2))

        if self.agv.angle < 0:
            agv_angle = self.agv.angle + 360.0
        else:
            agv_angle = self.agv.angle
        angle_offset = min(abs(agv_angle - TARGET_ANGLE), abs(360 - (agv_angle - TARGET_ANGLE)))
        # reward = 1 - 10 * distance / total_dis - 10 * angle_offset
        rate = distance / total_dis
        # print("rate: ", rate, " agv_x: ", self.agv.x, " agv_y: ", self.agv.y, " agv_angle: ", self.agv.angle)
        
        reward = 0
        if distance <= self.last_dis:
            reward += 10 * (self.last_dis - distance)
        else:
            reward -= 20 * (distance - self.last_dis)
        
        self.last_dis = distance
        
        if angle_offset >= 10:
            reward -=50
        
        if self.arriveGoal:
            reward += 200
        elif self.isCollide:
            reward -= 200
        elif self.angleOutofRange:
            reward -= 200
        
        print("rate: %.3f, agv_x: %.2f, agv_y: %.2f, agv_angle: %.3f, distance: %.3f, reward: %.3f" % (rate, self.agv.x, self.agv.y, self.agv.angle, distance, reward))
        # record_poistion_step_reward(self.agv.x, self.agv.y, reward)
        
        return reward
    
    def end(self):
        if math.sqrt(math.pow(self.agv.x - TARGET_X, 2) + math.pow(self.agv.y - TARGET_Y, 2)) <= 0.3:
            # arrive the goal point
            print("##########Arrive the goal point!##########")
            time.sleep(1)
            self.arriveGoal = True
            return True
        elif self.collisionDetect():
            # smash
            print("##########Smashed!##########")
            return True
        elif self.agv.angle >-75 or self.agv.angle < -110:
            self.angleOutofRange = True
            return True            
        else:
            return False
    
    def collisionDetect(self):
        widthOfMap = self.scene.gridMap.shape[1] * 0.05
        heightofMap = self.scene.gridMap.shape[0] * 0.05
        # print("widthOfMap: ", widthOfMap)
        # print("heightofMap: ", heightofMap)

        rowInGridMap = (heightofMap / 2 + self.agv.y) / 0.05
        colInGridMap = (widthOfMap / 2 + self.agv.x) / 0.05
        # print("self.agv.y: ", self.agv.y)
        # print("self.agv.x: ", self.agv.x)
        # print("agv rowInGridMap: ", rowInGridMap)
        # print("agv colInGridMap: ", colInGridMap)

        detectiveLineWidth, detectiveLineHeight = 10, 7
        indexWidthDetectiveLine_min = round(colInGridMap) - detectiveLineWidth
        indexWidthDetectiveLine_max = round(colInGridMap) + detectiveLineWidth
        indexHeightDetectiveLine_min = round(rowInGridMap) - detectiveLineHeight
        indexHeightDetectiveLine_max = round(rowInGridMap) + detectiveLineHeight
        # print("indexWidthDetectiveLine_min: ", indexWidthDetectiveLine_min)
        # print("indexWidthDetectiveLine_max: ", indexWidthDetectiveLine_max)
        # print("indexHeightDetectiveLine_min: ", indexHeightDetectiveLine_min)
        # print("indexHeightDetectiveLine_max: ", indexHeightDetectiveLine_max)

        detectiveArea = self.scene.gridMap[indexHeightDetectiveLine_min:indexHeightDetectiveLine_max+1, indexWidthDetectiveLine_min:indexWidthDetectiveLine_max+1]
        # print("self.scene.gridMap.shape: ",self.scene.gridMap.shape)
        # print("detectiveArea.shape: ", detectiveArea.shape)
        # print("detectiveArea: ", detectiveArea)

        obstacleIndices = np.argwhere(detectiveArea == 100)
        # print("obstacleIndices.shape: ", obstacleIndices.shape)
        if obstacleIndices.shape[0] >= 5:
            self.isCollide = True
        
        return self.isCollide

    def reset(self, pose_x, pose_y, angle):
        obs = self.simulator.simulate(self.agv, self.scene, RESET_ACTION, pose_x, pose_y, angle)
        self.arriveGoal = False
        self.isCollide = False
        self.angleOutofRange = False
        self.agv.lastAction = RESET_ACTION
        self.stepCount = 0
        self.stepReward = 0.
        self.totalReward = 1.0
        self.last_dis = math.sqrt(math.pow(AGV_START_POSITION_X - TARGET_X, 2) + math.pow(AGV_START_POSITION_Y - TARGET_X, 2))
        return obs


def main():
    env = RobotEnv()
    if not env.initRobotEnv():
        return
    else:
        print("RobotEnv init success!")
    
    env.reset()
    env.step(4)


    return


if __name__ == "__main__":
    main()
