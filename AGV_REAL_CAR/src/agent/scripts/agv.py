import os
import sys
path = os.path.abspath(".")
sys.path.insert(0,path + "/src/agent/scripts")

import numpy as np
from OpenGL.GL import *
import OpenGL.GL.shaders
import math


class AGV:
    def __init__(self):
        self.lengthOfBody = 0.76
        self.widthOfBody = 0.50
        self.radiusOfWheel = 0.07
        self.widthOfWheel = 0.06

        self.centerOfCircle = [0.0, 0.28]
        self.radiusOfCircle = 0.08
        self.slicesOfCircle = 18

        self.vertices = np.zeros(shape=[40, 6], dtype=np.float32)
        self.indices = np.zeros(shape=[28, 3], dtype=np.uint32)

        self.lastAction = -1
        self.x = 0.0
        self.y = 0.0
        self.angle = 0

      
    def calculateVerticesOfBody(self, position_x, position_y, theta):
        a = self.widthOfBody / 2
        b = self.lengthOfBody / 2

        verticesOfQuad = [-a,  -b, 0.0,  0.5, 0.5, 0.5,
                           a,  -b, 0.0,  0.5, 0.5, 0.5,
                           a,   b, 0.0,  0.5, 0.5, 0.5,
                          -a,   b, 0.0,  0.5, 0.5, 0.5]
        verticesOfQuad = np.array(verticesOfQuad, dtype=np.float32).reshape(4, 6)
        indiceOfQuad = [0, 1, 2,
                        2, 3, 0]
        indiceOfQuad = np.array(indiceOfQuad, dtype=np.uint32).reshape(2, 3)

        verticesOfBody = verticesOfQuad.copy()
        verticesOfBody[:, 0] = position_x + math.cos(theta) * verticesOfQuad[:, 0] - math.sin(theta) * verticesOfQuad[:, 1]
        verticesOfBody[:, 1] = position_y + math.sin(theta) * verticesOfQuad[:, 0] + math.cos(theta) * verticesOfQuad[:, 1]

        self.vertices[0:4, :] = verticesOfBody
        self.indices[0:2, :] = indiceOfQuad


    def calculateVerticesOfWheel(self, center_x, center_y, count, position_x, position_y, theta):
        a = center_x - self.widthOfWheel / 2
        b = center_y - self.radiusOfWheel
        c = center_x + self.widthOfWheel / 2
        d = center_y + self.radiusOfWheel

        if count < 3:
            verticesOfQuad = [a, b, 0.0,  0.0, 0.5, 0.0,
                              c, b, 0.0,  0.0, 0.5, 0.0,
                              c, d, 0.0,  0.0, 0.5, 0.0,
                              a, d, 0.0,  0.0, 0.5, 0.0]
        else:
            verticesOfQuad = [a, b, 0.0,  0.0, 0.0, 0.5,
                              c, b, 0.0,  0.0, 0.0, 0.5,
                              c, d, 0.0,  0.0, 0.0, 0.5,
                              a, d, 0.0,  0.0, 0.0, 0.5]
        verticesOfQuad = np.array(
            verticesOfQuad, dtype=np.float32).reshape(4, 6)
        indiceOfQuad = [0, 1, 2,
                        2, 3, 0]
        indiceOfQuad = np.array(indiceOfQuad, dtype=np.uint32).reshape(2, 3)
        indiceOfQuad = indiceOfQuad + 4 * count

        verticesOfWheel = verticesOfQuad.copy()
        verticesOfWheel[:, 0] = position_x + math.cos(
            theta) * verticesOfQuad[:, 0] - math.sin(theta) * verticesOfQuad[:, 1]
        verticesOfWheel[:, 1] = position_y + math.sin(
            theta) * verticesOfQuad[:, 0] + math.cos(theta) * verticesOfQuad[:, 1]

        self.vertices[4*count:4*count+4, :] = verticesOfWheel
        self.indices[2*count:2*count+2, :] = indiceOfQuad


    def calculateVerticesOfAGV(self, position_x, position_y, theta):
        self.calculateVerticesOfBody(position_x, position_y, theta)

        center = [0, 0]
        center[0] = -(self.widthOfBody + self.widthOfWheel) / 2
        center[1] = -self.lengthOfBody / 2 + self.radiusOfWheel
        self.calculateVerticesOfWheel(
            center[0], center[1], 1, position_x, position_y, theta)

        center_x = (self.widthOfBody + self.widthOfWheel) / 2
        center_y = -self.lengthOfBody / 2 + self.radiusOfWheel
        self.calculateVerticesOfWheel(
            center_x, center_y, 2, position_x, position_y, theta)

        center_x = -(self.widthOfBody + self.widthOfWheel) / 2
        center_y = self.lengthOfBody / 2 - self.radiusOfWheel
        self.calculateVerticesOfWheel(
            center_x, center_y, 3, position_x, position_y, theta)

        center_x = (self.widthOfBody + self.widthOfWheel) / 2
        center_y = self.lengthOfBody / 2 - self.radiusOfWheel
        self.calculateVerticesOfWheel(
            center_x, center_y, 4, position_x, position_y, theta)

        self.calculateVerticesOfCircle(position_x, position_y, theta)

        #print(self.vertices.shape)
        #print(self.indices.shape)

        #vertices = self.vertices.flatten()
        #indices = self.indices.flatten()

        #return vertices, indices


    def calculateVerticesOfCircle(self, position_x, position_y, theta):
        numberOfVertices = self.slicesOfCircle + 2
        doublePi = 2.0 * math.pi

        verticesX = np.zeros(shape=[numberOfVertices], dtype=np.float32)
        verticesY = np.zeros(shape=[numberOfVertices], dtype=np.float32)

        center = [0, 0]
        center[0] = position_x + math.cos(theta) * self.centerOfCircle[0] - \
            math.sin(theta) * self.centerOfCircle[1]
        center[1] = position_y + math.sin(theta) * self.centerOfCircle[0] + \
            math.cos(theta) * self.centerOfCircle[1]
        radius = self.radiusOfCircle

        verticesX[0] = center[0]
        verticesY[0] = center[1]
        #verticesZ[0] = 0

        for i in range(1, numberOfVertices):
            verticesX[i] = center[0] + radius * \
                math.cos((i-1) * doublePi / self.slicesOfCircle)
            verticesY[i] = center[1] + radius * \
                math.sin((i-1) * doublePi / self.slicesOfCircle)
            #print('{}, {}, {}'.format(i, verticesX[i], verticesY[i]))

        verticesInCircle = np.zeros(shape=[0, 6], dtype=np.float32)
        for i in range(0, numberOfVertices):
            vertice = [verticesX[i], verticesY[i], 0, 0.0, 0.0, 1.0]
            vertice = np.array(vertice, dtype=np.float32).reshape(1, 6)
            verticesInCircle = np.append(verticesInCircle, vertice, axis=0)

        firstIndex = np.max(self.indices) + 1

        indicesInCircle = np.zeros(shape=[0, 3], dtype=np.uint32)
        for i in range(0, self.slicesOfCircle):
            indice = [firstIndex, firstIndex+i+1, firstIndex+i+2]
            indice = np.array(indice, dtype=np.uint32).reshape(1, 3)
            indicesInCircle = np.append(indicesInCircle, indice, axis=0)

        #self.vertices = np.append(self.vertices, circleVertices, axis=0)
        #self.indices = np.append(self.indices, circleIndices, axis=0)

        self.vertices[20:40, :] = verticesInCircle
        self.indices[10:28, :] = indicesInCircle


    def initialize(self):
        self.calculateVerticesOfAGV(0.0, 0.0, 0.0)
        vertices = self.vertices.flatten()
        indices = self.indices.flatten()

        return vertices, indices


    def update(self, position_x, position_y, angle):
        theta = -angle * math.pi / 180.0
        self.calculateVerticesOfAGV(position_x, position_y, theta)
        vertices = self.vertices.flatten()
        indices = self.indices.flatten()

        return vertices, indices


    def getVertices(self):
        vertices = self.vertices.flatten()
        indices = self.indices.flatten()

        return vertices, indices


    def setPose(self, position_x, position_y, angle):
        self.x = position_x
        self.y = position_y
        self.angle = angle