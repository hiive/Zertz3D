from abc import ABC

from panda3d.core import TextureAttrib, AmbientLight, BitMask32
import numpy as np


class _BaseModel(ABC):
    def __init__(self, renderer, model_name, color=(1.0, 1.0, 1.0, 1.0)):
        self.model = renderer.loader.loadModel(model_name)
        self.model.reparentTo(renderer.render)
        self.model.setColor(color)
        self.model.setScale(0.3)

    def _set_pos(self, coord):
        self.model.setPos(coord)
        pass


class SkyBox(_BaseModel):
    def __init__(self, renderer):
        model_name = 'models/skybox.bam'
        super().__init__(renderer, model_name)
        self.model.setScale(16)
        self.model.setTwoSided(True)
        #self.model.setP(self.model, 8)
        self.model.setH(self.model, 15)
        self.model.setPos((0, 0, 5.9))
        self.model.setDepthWrite(False)
        self.model.hide(BitMask32(1))

        a_light = AmbientLight("a_sky_light")
        a_light.setColor((3.5, 3.5, 3.5, 1))
        a_node = renderer.render.attachNewNode(a_light)
        self.model.setLight(a_node)


class _BallBase(_BaseModel):
    def __init__(self, renderer, color):
        model_name = "models/ball_lo.bam"
        super().__init__(renderer, model_name, color)
        # self.model.flattenStrong()
        self._set_random_hpr()

    def _set_random_hpr(self):
        rot = np.random.uniform(low=0, high=360, size=(3,))
        self.model.setH(rot[0])
        self.model.setP(rot[1])
        self.model.setR(rot[2])

    def set_pos(self, coord, do_random_rotation=True):
        x, y, z = coord
        self._set_pos((x, y, z + 0.25))
        if do_random_rotation:
            self._set_random_hpr()


class BlackBallModel(_BallBase):
    def __init__(self, renderer):
        color = (0.25, 0.25, 0.25, 1)
        super().__init__(renderer, color)


class GrayBallModel(_BallBase):
    def __init__(self, renderer):
        color = (1.25, 1.25, 1.25, 1)
        super().__init__(renderer, color)


class WhiteBallModel(_BallBase):
    def __init__(self, renderer):
        color = (1.5, 1.5, 1.2, 1)
        super().__init__(renderer, color)
        for c in self.model.findAllMatches("**/+GeomNode"):
            gn = c.node()
            for i in range(gn.getNumGeoms()):
                state = gn.getGeomState(i)
                state = state.removeAttrib(TextureAttrib.getClassType())
                gn.setGeomState(i, state)

    def _set_random_hpr(self):
        pass


class BasePiece(_BaseModel):
    def __init__(self, renderer):
        color = (0.0, 0.0, 0.0, 1)
        model_name = "models/torus_lo.bam"

        super().__init__(renderer, model_name, color)
        self.model.setP(self.model, 180)
        # self.model.flattenStrong()

    def set_pos(self, coord):
        x, y, z = coord
        self._set_pos((x, y, z))
