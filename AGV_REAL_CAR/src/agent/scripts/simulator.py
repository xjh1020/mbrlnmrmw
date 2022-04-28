import os
import sys
path = os.path.abspath(".")
sys.path.insert(0,path + "/src/agent/scripts")

import glfw
import OpenGL.GL.shaders
import numpy as np
import matplotlib as plt
from scene import Scene
from agv import AGV
from gym import Env, spaces
from OpenGL.GL import *
from PIL import Image
import time
import math
import rospy

AGV_START_POSITION_X = 0.0
AGV_START_POSITION_Y = 0.0
AGV_START_ANGLE = -90.0

def ortho(left, right, bot, top, near, far):
    """ orthogonal projection matrix for OpenGL """
    dx, dy, dz = right - left, top - bot, far - near
    rx, ry, rz = -(right+left) / dx, -(top+bot) / dy, -(far+near) / dz
    return np.array([[2/dx, 0,    0,     rx],
                     [0,    2/dy, 0,     ry],
                     [0,    0,    -2/dz, rz],
                     [0,    0,    0,     1]], 'f')


def generate_shader():
    vertex_shader = """
    #version 330
    uniform mat4 MVP;
    in vec3 position;
    in vec3 color;

    out vec3 newColor;
    void main()
    {
        gl_Position = MVP * vec4(position, 1.0f);
        newColor = color;
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;

    out vec4 outColor;
    void main()
    {
        outColor = vec4(newColor, 1.0f);
    }
    """
    # print(glGetString(GL_VERSION))
    # print(glGetString(GL_SHADING_LANGUAGE_VERSION))
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    return shader


def drawTriangles(shader, vertices, indices):
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.itemsize *
                 len(vertices), vertices, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.itemsize *
                 len(indices), indices, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE,
                          vertices.itemsize * 6, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE,
                          vertices.itemsize * 6, ctypes.c_void_p(vertices.itemsize * 3))
    glEnableVertexAttribArray(color)

    glUseProgram(shader)


class Simulator:
    def __init__(self):
        self.widthOfWindow = None
        self.heightOfWindow = None
        self.shader = None
        self.mvp = None
        self.projMatrix = None

        self.verticesOfMap = None
        self.indicesOfMap = None
        self.verticesOfAGV = None
        self.indicesOfAGV = None

        self.vertices = None
        self.indices = None

        return


    def initialize(self, agv, scene):
        heightOfMap = scene.lengthOfMap
        widthOfMap = scene.widthOfMap
        self.verticesOfMap, self.indicesOfMap = scene.getVertices()
        maxIndiceOfMap = np.max(self.indicesOfMap)

        self.verticesOfAGV, self.indicesOfAGV = agv.initialize()
        self.indicesOfAGV = self.indicesOfAGV + maxIndiceOfMap + 1
        self.indices = np.append(self.indicesOfMap, self.indicesOfAGV)

        if not glfw.init():
            return False

        print("heightOfMap: ", heightOfMap)
        print("widthOfMap: ", widthOfMap)
    
        if heightOfMap >= widthOfMap:
            self.heightOfWindow = 512
            self.widthOfWindow = int((512 * widthOfMap) / heightOfMap)
        else:
            self.widthOfWindow = 512
            self.heightOfWindow = int((512 * heightOfMap) / widthOfMap)
        
        # print(self.heightOfWindow)
        # print(self.widthOfWindow)

        # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        # create display window

        self.window = glfw.create_window(
            self.widthOfWindow, self.heightOfWindow, "AGV State window", None, None)
    
        if not self.window:
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        
        # VAO = glGenVertexArrays(1)
        # glBindVertexArray(VAO)
        
        glViewport(0, 0, self.widthOfWindow, self.heightOfWindow)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        self.shader = generate_shader()
        self.mvp = glGetUniformLocation(self.shader, "MVP")
        self.projMatrix = ortho(-widthOfMap/2, widthOfMap/2, -heightOfMap/2, heightOfMap/2, -1, 1)

        return True


    def calcalateNextPoseOfAGV(self, agv, action):
        position_x = agv.x
        position_y = agv.y
        angle = agv.angle

        # xu add --- record action
        # agv.lastAction = action
        p_x = 0.
        p_y = 0.

        if action == 0:
            p_x = np.random.normal(loc=0.1, scale=3.1456e-7)
            p_y = 0.
        elif action == 1:
            p_x = np.random.normal(loc=-0.1, scale=2.9185e-07)
            p_y = 0.
        elif action == 2:
            p_x = 0.
            p_y = np.random.normal(loc=0.1, scale=1.7808e-07)
        elif action == 3:
            p_x = 0.
            p_y = np.random.normal(loc=-0.1, scale=1.7878e-07)
        elif action == 4:
            angle = agv.angle + 5
        elif action == 5:
            angle = agv.angle - 5
        
        delta_x = p_x * math.cos(math.radians(agv.angle)) + p_y * math.sin(math.radians(agv.angle))
        delta_y = -p_x * math.sin(math.radians(agv.angle)) + p_y * math.cos(math.radians(agv.angle))

        # print("delta_x: ", delta_x)
        # print("delta_y: ", delta_y)

        position_x = agv.x + delta_x
        position_y = agv.y + delta_y

        if action == -1:
            position_x = AGV_START_POSITION_X
            position_y = AGV_START_POSITION_Y
            angle = AGV_START_ANGLE

        return position_x, position_y, angle


    def saveImage(self, win, position_x, position_y, angle):
        width, height = glfw.get_framebuffer_size(win)

        nrChannels = 3
        stride = nrChannels * width
        stride += (4 - stride % 4) if (stride % 4) else 0
        bufferSize = stride * height

        # time.sleep(1)
        glPixelStorei(GL_PACK_ALIGNMENT, 4)
        
        # glReadBuffer(GL_BACK)
        glReadBuffer(GL_FRONT)  # xu update

        
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        filename = "./image/{:.2f}_{:.2f}_{:.2f}.png".format(position_x, position_y, angle)
        # image.save("./image.png")
        # image.save(filename)
        im = np.array(image)
        # print(im)
        # print(im.shape)
        return im  # numpy


    def simulate(self, agv, scene, action, pose_x, pose_y, th):
        # position_x, position_y, angle = self.calcalateNextPoseOfAGV(agv, action)
        position_x, position_y, angle = pose_x, pose_y, th
        
        if action == -1:
            position_x = AGV_START_POSITION_X
            position_y = AGV_START_POSITION_Y
            angle = AGV_START_ANGLE

        if not glfw.window_should_close(self.window):
            glfw.poll_events()
            glClear(GL_COLOR_BUFFER_BIT)

            self.verticesOfAGV, _ = agv.update(position_x, position_y, angle)
            self.vertices = np.append(self.verticesOfMap, self.verticesOfAGV)
            #indices = np.append(indicesOfMap, indicesOfAGV, axis=0)
            drawTriangles(self.shader, self.vertices, self.indices)
            #angle = angle + 1.0

            glUniformMatrix4fv(self.mvp, 1, GL_FALSE, self.projMatrix)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
            glfw.swap_buffers(self.window)

            agv.setPose(position_x, position_y, angle)
            # time.sleep(0.1)  # xu add
            obs = self.saveImage(self.window, position_x, position_y, angle)
            return obs  # numpy (314, 512, 3)


    def terminate(self):
        glfw.terminate()


        

        

