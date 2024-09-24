from panda3d.core import CardMaker, Texture, Plane, PlaneNode, TransparencyAttrib, \
    CullFaceAttrib, RenderState, Shader, TextureStage
from panda3d.core import Vec4
from direct.showbase.ShowBase import ShowBase


class WaterNode:

    def __init__(self, renderer, x1, y1, x2, y2, z, anim, distort):
        # anim: vx, vy, scale, skip
        # distort: offset, strength, refraction factor (0 = perfect mirror,
        #   1 = total refraction), refractivity
        self.renderer = renderer

        # Create a texture buffer for reflection
        self.buffer = renderer.win.makeTextureBuffer('waterBuffer', 1024, 1024)
        self.buffer.setClearColor((0, 0, 0, 1))  # Clear to opaque black
        # self.buffer.setSort(1)  # Ensure it renders after the main scene

        # Create a camera for rendering reflection
        self.water_cam_np = renderer.makeCamera(self.buffer)
        self.water_cam_np.reparentTo(renderer.render)

        # Configure the reflection camera's lens
        cam = self.water_cam_np.node()
        cam.getLens().setFov(renderer.camLens.getFov())
        cam.getLens().setNearFar(1, 5000)
        cam.setInitialState(RenderState.make(CullFaceAttrib.makeReverse()))
        cam.setTagStateKey('Clipped')

        # Create the water surface geometry
        maker = CardMaker('water')  # Water surface
        maker.setFrame(x1, x2, y1, y2)

        self.waterNP = renderer.render.attachNewNode(maker.generate())
        self.waterNP.setPosHpr((0, 0, z), (0, -90, 0))
        self.waterNP.setTransparency(TransparencyAttrib.MAlpha)

        # Load and set the GLSL shader
        shader = Shader.load(Shader.SL_GLSL, vertex="shaders/water.vert.glsl", fragment="shaders/water.frag.glsl")
        if not shader:
            raise RuntimeError("Failed to load GLSL shader.")


        self.waterNP.setShader(shader)

        # Set shader inputs
        self.waterNP.setShaderInput('k_wateranim', anim)
        self.waterNP.setShaderInput('k_waterdistort', distort)
        self.waterNP.setShaderInput('k_time', 0.0)  # Initialize time to 0

        # Define the reflection plane
        self.waterPlane = Plane((0, 0, z + 1), (0, 0, z))  # Reflection plane
        plane_node = PlaneNode('waterPlane')
        plane_node.setPlane(self.waterPlane)
        self.waterNP.attachNewNode(plane_node)

        # Load textures
        tex0 = self.buffer.getTexture()  # Reflection texture, created in realtime by the 'water camera'
        tex0.setWrapU(Texture.WMClamp)
        tex0.setWrapV(Texture.WMClamp)

        tex1 = self.renderer.loader.loadTexture('models/source.png')  # Distortion texture
        if not tex1:
            raise RuntimeError("Failed to load distortion texture 'models/source.png'")

        # self.waterNP.setTexture(TextureStage('reflection'), tex0)
        # self.waterNP.setTexture(TextureStage('distortion'), tex1)  # distortion texture

        # Bind textures directly to shader uniforms
        self.waterNP.setShaderInput('tex_0', tex0)  # Reflection Texture
        self.waterNP.setShaderInput('tex_1', tex1)  # Distortion Texture

        # Add the update task
        self.task = self.renderer.taskMgr.add(self.update, 'waterUpdate', sort=50)

    def remove(self):
        self.waterNP.removeNode()
        self.renderer.taskMgr.remove(self.task)

    def destroy(self):
        self.renderer.graphicsEngine.removeWindow(self.buffer)
        self.renderer.win.removeDisplayRegion(
            self.water_cam_np.node().getDisplayRegion(0))
        self.water_cam_np.removeNode()

    def update(self, task):
        # Update shader inputs
        self.waterNP.setShaderInput('k_time', task.time)  # Update time uniform

        # Update the reflection camera's matrix
        mx = self.renderer.camera.getMat() * self.waterPlane.getReflectionMat()
        self.water_cam_np.setMat(mx)  # Update matrix of the reflection camera

        return task.cont