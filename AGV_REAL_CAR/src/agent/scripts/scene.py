import os
import sys
path = os.path.abspath(".")
sys.path.insert(0,path + "/src/agent/scripts")

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import math


sizeOfCell = 0.05
startCoordinate = [0.0, 0.0]


class Scene:
    def __init__(self):
        #self.numberOfCells = (int)(np.maximum((height / sizeOfCell), (width / sizeOfCell)))
        #self.gridMap = np.zeros(self.numberOfCells * self.numberOfCells).reshape(self.numberOfCells, self.numberOfCells)
        self.lengthOfMap = None
        self.widthOfMap = None
        self.targetX = None
        self.targetY = None

        self.vertices = np.empty(shape=[0, 6], dtype=np.float32)
        self.indices = np.empty(shape=[0, 3], dtype=np.uint32)


    def loadRosGridMap(self):
        self.gridMap[20:32, 20:22] = 100.0
        self.scale = 1.0
        self.generateObstacleVertices()

        vertices = self.vertices.flatten()
        indices = self.indices.flatten()

        return vertices, indices


    def loadRawGripMap(self, filename):
        file = open(filename)
        lines = file.readlines()
        rawMap = np.empty(shape=[0, 1984], dtype=np.float32)

        for line in lines:
            # print(line)
            data = np.fromstring(line, dtype=float, sep=',')
            rawMap = np.append(rawMap, data.reshape(1, 1984), axis=0)
        file.close()

        self.clipMap(rawMap)
        self.generateObstacleVertices()

        return self.lengthOfMap, self.widthOfMap


    def clipMap(self, rawMap):
        indexs = np.argwhere(rawMap == 100)
        maxRow, maxCol = np.max(indexs, axis=0)
        minRow, minCol = np.min(indexs, axis=0)
        rows = maxRow - minRow + 1
        cols = maxCol - minCol + 1

        self.rows = rows
        self.cols = cols
        self.lengthOfMap = rows * 0.05
        self.widthOfMap = cols * 0.05

        startCoordinate[0] = -self.widthOfMap/2
        startCoordinate[1] = -self.lengthOfMap/2

        self.gridMap = np.zeros(rows * cols).reshape(rows, cols)
        self.gridMap[0:rows, 0:cols] = rawMap[minRow:maxRow+1, minCol:maxCol+1]
        return

        #self.gridMap[[0, rows-1], :] = 100
        #self.gridMap[:, [0, cols-1]] = 100


    def generateObstacleVertices(self):
        verticesOfQuad = [0.0,  0.0, 0.0,  1.0, 1.0, 1.0,
                          0.05, 0.0, 0.0,  1.0, 1.0, 1.0,
                          0.05, 0.05, 0.0,  1.0, 1.0, 1.0,
                          0.0,  0.05, 0.0,  1.0, 1.0, 1.0]
        verticesOfQuad = np.array(
            verticesOfQuad, dtype=np.float32).reshape(4, 6)
        indiceOfQuad = [0, 1, 2,
                        2, 3, 0]
        indiceOfQuad = np.array(indiceOfQuad, dtype=np.uint32).reshape(2, 3)

        indexs = np.argwhere(self.gridMap == 100.0)
        count = 0

        for index in indexs:
            obstacleCellY = startCoordinate[1] + (index[0] * 0.05)
            obstacleCellX = startCoordinate[0] + (index[1] * 0.05)

            verticesOfObstacleCell = verticesOfQuad.copy()
            verticesOfObstacleCell[:, 0] = verticesOfObstacleCell[:, 0] + obstacleCellX
            verticesOfObstacleCell[:, 1] = verticesOfObstacleCell[:, 1] + obstacleCellY
            # print(temp_quad)

            indiceOfObstacleCell = indiceOfQuad.copy()
            indiceOfObstacleCell = indiceOfObstacleCell + 4 * count

            #self.quads = np.append(self.quads, temp_quad.reshape(1, 24), axis=0)
            #self.indices = np.append(self.indices, temp_indice.reshape(1, 6), axis=0)
            self.vertices = np.append(self.vertices, verticesOfObstacleCell, axis=0)
            self.indices = np.append(self.indices, indiceOfObstacleCell, axis=0)
            count = count + 1


    def getVertices(self):
        #self.generateObstacleVertices()
        vertices = self.vertices.flatten()
        indices = self.indices.flatten()

        return vertices, indices


    def generateVerticesOfTarget(self):
        radius = 0.1
        slicesOfCircle = 18
        numberOfVertices = slicesOfCircle + 2
        doublePi = 2.0 * math.pi

        verticesX = np.zeros(shape=[numberOfVertices], dtype=np.float32)
        verticesY = np.zeros(shape=[numberOfVertices], dtype=np.float32)

        center = [0, 0]
        center[0] = self.targetX
        center[1] = self.targetY

        verticesX[0] = center[0]
        verticesY[0] = center[1]

        for i in range(1, numberOfVertices):
            verticesX[i] = center[0] + radius * \
                math.cos((i-1) * doublePi / slicesOfCircle)
            verticesY[i] = center[1] + radius * \
                math.sin((i-1) * doublePi / slicesOfCircle)

        verticesInCircle = np.zeros(shape=[0, 6], dtype=np.float32)
        for i in range(0, numberOfVertices):
            vertice = [verticesX[i], verticesY[i], 0, 1.0, 0.0, 0.0]
            vertice = np.array(vertice, dtype=np.float32).reshape(1, 6)
            verticesInCircle = np.append(verticesInCircle, vertice, axis=0)

        firstIndex = np.max(self.indices) + 1

        indicesInCircle = np.zeros(shape=[0, 3], dtype=np.uint32)
        for i in range(0, slicesOfCircle):
            indice = [firstIndex, firstIndex+i+1, firstIndex+i+2]
            indice = np.array(indice, dtype=np.uint32).reshape(1, 3)
            indicesInCircle = np.append(indicesInCircle, indice, axis=0)

        self.vertices = np.append(self.vertices, verticesInCircle, axis=0)
        self.indices = np.append(self.indices, indicesInCircle, axis=0)

    
    def setTarget(self, position_x, position_y):
        self.targetX = position_x
        self.targetY = position_y
        self.generateVerticesOfTarget()