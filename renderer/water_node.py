from panda3d.core import CardMaker, Texture, TextureStage, Plane, \
    PlaneNode, TransparencyAttrib, CullFaceAttrib, RenderState, ShaderAttrib


class WaterNode:

    def __init__(self, render, x1, y1, x2, y2, z, anim, distort):
        # anim: vx, vy, scale, skip
        # distort: offset, strength, refraction factor (0 = perfect mirror,
        #   1 = total refraction), refractivity
        self.render = render

        self.buffer = render.win.makeTextureBuffer('waterBuffer', 1024, 1024)
        self.water_cam_np = render.makeCamera(self.buffer)

        maker = CardMaker('water')  # Water surface
        maker.setFrame(x1, x2, y1, y2)

        self.waterNP = render.render.attachNewNode(maker.generate())
        self.waterNP.setPosHpr((0, 0, z), (0, -90, 0))
        self.waterNP.setTransparency(TransparencyAttrib.MAlpha)
        self.waterNP.setShader(self.render.loader.loadShader('shaders/water.sha'))
        self.waterNP.setShaderInput('wateranim', anim)
        self.waterNP.setShaderInput('waterdistort', distort)
        self.waterNP.setShaderInput('time', 0)

        self.waterPlane = Plane((0, 0, z + 1), (0, 0, z))  # Reflection plane
        PlaneNode('waterPlane').setPlane(self.waterPlane)

        self.buffer.setClearColor((0, 0, 0, 1))  # buffer

        self.water_cam_np.reparentTo(render.render)  # reflection camera
        cam = self.water_cam_np.node()
        cam.getLens().setFov(self.render.camLens.getFov())
        cam.getLens().setNearFar(1, 5000)
        cam.setInitialState(RenderState.make(CullFaceAttrib.makeReverse()))
        cam.setTagStateKey('Clipped')
        """
        cam.setTagState('True', RenderState.make(
            ShaderAttrib.make().setShader(
                self.render.loader.loadShader('shaders/splut3Clipped.sha'))))
        """

        tex0 = self.buffer.getTexture()  # reflection texture, created in
        # realtime by the 'water camera'
        tex0.setWrapU(Texture.WMClamp)
        tex0.setWrapV(Texture.WMClamp)
        tex1 = self.render.loader.loadTexture('models/source.png')

        self.waterNP.setTexture(TextureStage('reflection'), tex0)
        self.waterNP.setTexture(TextureStage('distortion'), tex1)  # distortion texture

        self.task = self.render.taskMgr.add(self.update, 'waterUpdate', sort=50)

    def remove(self):
        self.waterNP.removeNode()
        self.render.taskMgr.remove(self.task)

    def destroy(self):
        self.render.graphicsEngine.removeWindow(self.buffer)
        self.render.win.removeDisplayRegion(
            self.water_cam_np.node().getDisplayRegion(0))
        self.water_cam_np.removeNode()

    def update(self, task):
        self.waterNP.setShaderInput('time', task.time)  # time 4 H2O distortions
        mx = self.render.camera.getMat() * self.waterPlane.getReflectionMat()
        self.water_cam_np.setMat(mx)  # update matrix of the reflection camera
        return task.cont