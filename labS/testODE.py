from panda3d.ode import OdeWorld
from panda3d.ode import OdeBody, OdeMass
from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
import numpy as np
from panda3d.core import ClockObject
import panda3d.core as pc
from panda3d.core import TextNode, TransparencyAttrib
from panda3d.core import LPoint3, LVector3
from panda3d.core import SamplerState
from direct.gui.OnscreenText import OnscreenText

class testViewer(ShowBase):
    myWorld = OdeWorld()
    myWorld.setGravity(0, 0, -9.81)

    myBody = OdeBody(myWorld)
    # myBody.setPosition(somePandaObject.getPos(render))
    # myBody.setQuaternion(somePandaObject.getQuat(render))
    # myBody.setPosition()
    # myBody.setQuaternion(somePandaObject.getQuat(render))
    myMass = OdeMass()
    myMass.setBox(11340, 1, 1, 1)
    myBody.setMass(myMass)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model",
                                {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)
        # Loop its animation.
        self.pandaActor.loop("walk")

        # Create the four lerp intervals needed for the panda to
        # walk back and forth.
        posInterval1 = self.pandaActor.posInterval(13,
                                                   Point3(0, -10, 0),
                                                   startPos=Point3(0, 10, 0))
        posInterval2 = self.pandaActor.posInterval(13,
                                                   Point3(0, 10, 0),
                                                   startPos=Point3(0, -10, 0))
        hprInterval1 = self.pandaActor.hprInterval(3,
                                                   Point3(180, 0, 0),
                                                   startHpr=Point3(0, 0, 0))
        hprInterval2 = self.pandaActor.hprInterval(3,
                                                   Point3(0, 0, 0),
                                                   startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        self.pandaPace = Sequence(posInterval1, hprInterval1,
                                  posInterval2, hprInterval2,
                                  name="pandaPace")
        self.pandaPace.loop()


    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


# app = MyApp()
# app.run()

# https://blog.csdn.net/AI_LX/article/details/116305104

def create_link(self, link_id, position, scale, rot):
    # create a link
    box = self.loader.loadModel("material/GroundScene.egg")
    # node = self.render.attachNewNode(f"link{link_id}")
    node = self.render.attachNewNode(f"ttt{link_id}")
    box.reparentTo(node)
    #
    # # add texture
    box.setTextureOff(1)
    box.setTexture(self.tex, 1)
    box.setScale(*scale)

    node.setPos(self.render, *position)
    if rot is not None:
        node.setQuat(self.render, pc.Quat(*rot[[3, 0, 1, 2]].tolist()))
    return node

SPRITE_POS = 55     # At default field of view and a depth of 55, the screen
def loadObject(tex=None, pos=LPoint3(0, 0), depth=SPRITE_POS, scale=1,
               transparency=True):
    # Every object uses the plane model and is parented to the camera
    # so that it faces the screen.
    mode = 2
    if mode == 1:
        # obj = loader.loadModel("models/plane")
        obj = loader.loadModel("material/GroundScene.egg")
        obj.reparentTo(camera)

        # Set the initial position and scale.
        obj.setPos(pos.getX(), depth, pos.getY())
        obj.setScale(scale)

        # This tells Panda not to worry about the order that things are drawn in
        # (ie. disable Z-testing).  This prevents an effect known as Z-fighting.
        obj.setBin("unsorted", 0)
        obj.setDepthTest(False)

    elif mode == 2:
        ground = loader.loadModel("material/GroundScene.egg")
        # ground.reparentTo(render)
        ground.reparentTo(camera)
        ground.setPos(pos.getX(), depth, pos.getY())
        ground.setScale(scale)
        ground.setTexScale(pc.TextureStage.getDefault(), 5, 5)

        ground.setBin("unsorted", 0)
        ground.setDepthTest(False)

    # if transparency:
    #     # Enable transparency blending.
    #     obj.setTransparency(TransparencyAttrib.MAlpha)

    if tex:
        if False:
            # Load and set the requested texture.
            tex = loader.loadTexture("textures/" + tex)
            tex.setWrapU(SamplerState.WM_clamp)
            tex.setWrapV(SamplerState.WM_clamp)
            if mode == 1:
                obj.setTexture(tex, 1)
            elif mode == 2:
                ground.setTexture(tex, 1)


class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # self.disableMouse()

        properties = WindowProperties()
        properties.setSize(1000, 750)
        self.win.requestProperties(properties)

        self.setBackgroundColor((0, 0, 0, 1))
        # self.bg = loadObject("stars.jpg", scale=146, depth=200,
        #                      transparency=False)

        self.bg = loadObject("stars.jpg", scale=50, depth=200,
                             transparency=False)

        # self.ground = self.loader.loadModel("material/GroundScene.egg")
        # self.ground.reparentTo(self.render)
        # self.ground.setScale(100, 1, 100)
        # self.ground.setTexScale(pc.TextureStage.getDefault(), 50, 50)
        # self.ground.setPos(0, -1, 0)

        # self.camera.setPos(0, 0, 32)
        # self.camera.setP(-90)

def testOde():
    physics_body = []
    physics_joint = []
    damping_joint = []
    world = OdeWorld()
    world.setGravity(0, -9.81, 0)
    #self.world.set_cfm(0.00001)
    world.initSurfaceTable(2)
    world.setSurfaceEntry(0, 0, 1.5, 0.0, 0, 0.9, 0.001, 0.0, 0.000)

    # space = OdeSimpleSpace()
    # contactgroup = OdeJointGroup()
    # space.setAutoCollideWorld(self.world)
    # space.setAutoCollideJointGroup(self.contactgroup)
    # groundGeom = OdePlaneGeom(self.space, pc.Vec4(0, 1, 0, 0))
    # groundGeom.setCollideBits(pc.BitMask32(0x00000001))
    # groundGeom.setCategoryBits(pc.BitMask32(0x00000001))


    # ode_body = OdeBody(world)
    # ode_body.setPosition(body[-1].getPos(self.render))
    # ode_body.setQuaternion(body[-1].getQuat(self.render))
    # mass = OdeMass()
    # mass.setBox(100, *[ j * 10 for j in scale[i]])
    # ode_body.setMass(mass)


    myBody = OdeBody(world)
    myBody.setPosition((0,2,0))
    # myBody.setQuaternion(somePandaObject.getQuat(render))
    myMass = OdeMass()
    # myMass.setBox(11340, 1, 1, 1)
    myMass.setBox(1000, 1, 1, 1)
    myBody.setMass(myMass)
    myBody.addForce( (2000,0,0) )

    # Do the simulation...
    total_time = 0.0
    dt = 0.04
    while total_time<2.0:
        x,y,z = myBody.getPosition()
        u,v,w = myBody.getLinearVel()
        print("%1.2fsec: pos=(%6.3f, %6.3f, %6.3f)  vel=(%6.3f, %6.3f, %6.3f)" % \
              (total_time, x, y, z, u,v,w))
        world.step(dt)
        total_time+=dt


testMode = 2
if testMode == 1:
    game = Game()
    game.run()
elif testMode == 2:
    testOde()