from panda3d.core import (
    CardMaker,
    Texture,
    Plane,
    PlaneNode,
    TransparencyAttrib,
    CullFaceAttrib,
    RenderState,
    Shader,
    BitMask32,
)


class WaterNode:
    def __init__(self, renderer, x1, y1, x2, y2, z, anim, distort):
        # anim: vx, vy, scale, skip
        # distort: offset, strength, refraction factor (0 = perfect mirror,
        #   1 = total refraction), refractivity
        self.renderer = renderer

        # Create a texture buffer for reflection matching screen aspect ratio
        screen_x = renderer.win.getXSize()
        screen_y = renderer.win.getYSize()
        self.buffer = renderer.win.makeTextureBuffer("waterBuffer", screen_x, screen_y)
        self.buffer.setClearColor((0.5, 0.7, 1.0, 1))  # Clear to sky blue for debugging
        # self.buffer.setSort(1)  # Ensure it renders after the main scene

        # CRITICAL: Enable simplepbr for this buffer so objects render correctly
        # The buffer needs to inherit the shader generator from the main window
        self.buffer.setOneShot(False)
        self.buffer.setActive(True)

        # Create a camera for rendering reflection
        self.water_cam_np = renderer.makeCamera(self.buffer)
        self.water_cam_np.reparentTo(renderer.render)

        # Configure the reflection camera's lens
        cam = self.water_cam_np.node()
        # cam.getLens().setFov(renderer.camLens.getFov())
        # cam.getLens().setNearFar(1, 5000)
        reflection_lens = self.renderer.camLens.makeCopy()
        cam.setLens(reflection_lens)

        # Set camera mask to render all normal objects
        # BitMask32.bit(0) is the default rendering mask
        cam.setCameraMask(BitMask32.bit(0))

        cam.setInitialState(RenderState.make(CullFaceAttrib.makeReverse()))
        cam.setTagStateKey("Clipped")

        # Create the water surface geometry
        maker = CardMaker("water")  # Water surface
        maker.setFrame(x1, x2, y1, y2)

        self.waterNP = renderer.render.attachNewNode(maker.generate())
        self.waterNP.setPos(
            0, 0, z - 0.1
        )  # Position slightly below z=0 to render behind board
        self.waterNP.setTransparency(TransparencyAttrib.MAlpha)

        # Disable depth write so water doesn't block objects in front
        self.waterNP.setDepthWrite(True)

        # Load and set the GLSL shader
        shader = Shader.load(
            Shader.SL_GLSL,
            vertex="shaders/water.vert.glsl",
            fragment="shaders/water.frag.glsl",
        )
        if not shader:
            raise RuntimeError("Failed to load GLSL shader.")

        # CRITICAL: Disable shader auto-generation FIRST, then set custom shader
        self.waterNP.setShaderAuto(False)
        self.waterNP.setShader(shader, 1)  # Priority 1 to override any other shaders

        # Set shader inputs
        self.waterNP.setShaderInput("k_wateranim", anim)
        self.waterNP.setShaderInput("k_waterdistort", distort)
        self.waterNP.setShaderInput("k_time", 0.0)  # Initialize time to 0

        # Pass screen size to shader
        from panda3d.core import LVector2f

        screen_size = LVector2f(renderer.win.getXSize(), renderer.win.getYSize())
        self.waterNP.setShaderInput("k_screen_size", screen_size)

        # Get model-view matrix (from model to camera space) for main camera
        mat_modelview = self.waterNP.getMat(self.renderer.render)

        # Get projection matrix from the main camera's lens
        mat_projection = self.renderer.camLens.getProjectionMat()

        # Compute model-view-projection matrix for main camera
        mat_modelproj = mat_projection * mat_modelview

        # Get the model matrix (water surface to world transform)
        model_matrix = self.waterNP.getMat(self.renderer.render)

        # Set shader inputs (k_reflection_viewproj will be updated in the update task)
        self.waterNP.setShaderInput(
            "k_reflection_viewproj", mat_modelproj
        )  # Initialize, will be updated
        self.waterNP.setShaderInput(
            "p3d_ModelMatrix", model_matrix
        )  # Pass model matrix to shader

        # Define the reflection plane
        # self.waterPlane = Plane((0, 0, z + 1), (0, 0, z))  # Reflection plane
        self.waterPlane = Plane((0, 0, 1), (0, 0, z))  # Reflection plane
        self.waterNP.setPosHpr((0, 0, z), (0, 0, 0))  # No rotation
        plane_node = PlaneNode("waterPlane")
        plane_node.setPlane(self.waterPlane)
        self.waterNP.attachNewNode(plane_node)

        # Load textures
        tex0 = (
            self.buffer.getTexture()
        )  # Reflection texture, created in realtime by the 'water camera'
        tex0.setWrapU(Texture.WMClamp)
        tex0.setWrapV(Texture.WMClamp)

        tex1 = self.renderer.loader.loadTexture(
            "models/source.png"
        )  # Distortion texture
        if not tex1:
            raise RuntimeError("Failed to load distortion texture 'models/source.png'")

        # self.waterNP.setTexture(TextureStage('reflection'), tex0)
        # self.waterNP.setTexture(TextureStage('distortion'), tex1)  # distortion texture

        # Bind textures directly to shader uniforms
        self.waterNP.setShaderInput("tex_0", tex0)  # Reflection Texture
        self.waterNP.setShaderInput("tex_1", tex1)  # Distortion Texture

        # self.waterNP.clearShader()
        # DEBUG: After creating the WaterNode instance
        def create_reflection_debug_card(tex):
            cm = CardMaker("reflection_debug_card")
            # cm.setFrame(-10, 10, -1, 1)  # Set the size of the card
            cm.setFrame(x1, x2, y1, y2)
            reflection_card_np = self.renderer.aspect2d.attachNewNode(cm.generate())
            reflection_card_np.setTexture(tex)
            reflection_card_np.setTransparency(TransparencyAttrib.MAlpha)
            reflection_card_np.setScale(0.15)
            reflection_card_np.setPos(0, 0, -0.2)
            return reflection_card_np

        # reflection_debug_card = create_reflection_debug_card(tex0)

        # DEBUG: Create a display region to show the reflection camera's view
        # reflection_dr = self.renderer.win.makeDisplayRegion(0.7, 1.0, 0.7, 1.0)
        # reflection_dr.setCamera(self.water_cam_np)
        # reflection_dr.setSort(20)

        # self.waterNP.clearShader()
        # Add the update task
        self.task = self.renderer.taskMgr.add(self.update, "waterUpdate", sort=50)

    def remove(self):
        self.waterNP.removeNode()
        self.renderer.taskMgr.remove(self.task)

    def destroy(self):
        self.renderer.graphicsEngine.removeWindow(self.buffer)
        self.renderer.win.removeDisplayRegion(
            self.water_cam_np.node().getDisplayRegion(0)
        )
        self.water_cam_np.removeNode()

    def update(self, task):
        # Update shader inputs
        self.waterNP.setShaderInput("k_time", task.time)  # Update time uniform

        # Get the main camera's NodePath
        main_cam_np = self.renderer.cam

        # Get the main camera's transformation matrix relative to render
        main_cam_mat = main_cam_np.getMat(self.renderer.render)

        # Get the reflection matrix based on the water plane
        reflection_mat = self.waterPlane.getReflectionMat()

        # Calculate the reflection camera's matrix
        # reflection_cam_mat = reflection_mat * main_cam_mat
        reflection_cam_mat = main_cam_mat * reflection_mat
        self.water_cam_np.setMat(
            reflection_cam_mat
        )  # Update matrix of the reflection camera

        # Calculate the reflection camera's view-projection matrix for texture sampling
        # This needs to transform vertices from world space to reflection camera's clip space

        # Get the reflection camera's view matrix (world to camera space)
        reflection_view_mat = self.water_cam_np.getMat(self.renderer.render)
        reflection_view_mat.invertInPlace()

        # Get projection matrix from the reflection camera's lens
        reflection_proj_mat = self.water_cam_np.node().getLens().getProjectionMat()

        # Compute the complete matrix: projection * view
        # This transforms from world space to reflection camera's clip space
        mat_reflection_proj = reflection_proj_mat * reflection_view_mat

        # Update shader input with the reflection camera's projection matrix
        self.waterNP.setShaderInput("k_reflection_viewproj", mat_reflection_proj)

        # Debugging: Print camera positions
        # main_cam_pos = main_cam_np.getPos(self.renderer.render)
        # reflection_cam_pos = self.water_cam_np.getPos(self.renderer.render)
        # print(f"Main Camera Position: {main_cam_pos}")
        # print(f"Reflection Camera Position: {reflection_cam_pos}")
        return task.cont
