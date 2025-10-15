from abc import ABC
import random

from panda3d.core import TextureAttrib, AmbientLight, BitMask32


class _BaseModel(ABC):
    def __init__(self, renderer, model_name, color=(1.0, 1.0, 1.0, 1.0)):
        self.renderer = renderer
        self.model_name = model_name
        self.color = color
        self.pos_coords = None
        self.saved_coords = None
        self.model = None
        self.add()

    def _set_pos(self, coord):
        self.pos_coords = coord
        self.model.setPos(self.pos_coords)
        pass

    def get_pos(self):
        return self.pos_coords

    def get_scale(self):
        return self.model.getScale()

    def remove(self):
        if self.model is not None:
            self.model.removeNode()
            self.model = None

    def _set_collide_mask(self, mask):
        if self.model is None:
            return
        self.model.setCollideMask(mask)
        for node in self.model.findAllMatches("**"):
            node.setCollideMask(mask)

    def add(self):
        if self.model is not None:
            return
        # self.model.reparentTo(None)
        self.model = self.renderer.loader.loadModel(self.model_name)
        self.model.setColor(self.color)
        # self.model.setScale(0.3)
        self.model.setScale(0.305)
        self.model.reparentTo(self.renderer.render)
        self._set_collide_mask(BitMask32.allOff())

    def show(self):
        if self.model is not None:
            self.model.show()

    def hide(self):
        if self.model is not None:
            self.model.hide()


class SkyBox(_BaseModel):
    def __init__(self, renderer):
        model_name = "models/skybox.bam"
        super().__init__(renderer, model_name)
        self.model.setScale(16)  # Increased from 16 to zoom out
        self.model.setTwoSided(True)
        # self.model.setP(self.model, 8)
        self.model.setH(self.model, 15)
        self.model.setPos((0, 0, 4.9))
        self.model.setDepthWrite(False)
        self.model.hide(BitMask32(1))

        a_light = AmbientLight("a_sky_light")
        a_light.setColor((3.5, 3.5, 3.5, 1))
        a_node = renderer.render.attachNewNode(a_light)
        self.model.setLight(a_node)


def make_marble(renderer, color):
    if color == "w":
        return WhiteBallModel(renderer)
    elif color == "b":
        return BlackBallModel(renderer)
    elif color == "g":
        return GrayBallModel(renderer)
    return None


class _BallBase(_BaseModel):
    def __init__(self, renderer, color):
        model_name = "models/ball_lo.bam"
        super().__init__(renderer, model_name, color)
        # self.model.flattenStrong()
        # Each marble gets its own unseeded RNG for visual randomness
        # This keeps marble rotations independent from game logic RNG
        self.rng = random.Random()
        self._set_random_hpr()
        self.z_offset = 0.25
        self._set_collide_mask(BitMask32.bit(1))

    def _set_random_hpr(self):
        h = self.rng.uniform(0, 360)
        p = self.rng.uniform(0, 360)
        r = self.rng.uniform(0, 360)
        self.model.setH(h)
        self.model.setP(p)
        self.model.setR(r)

    def set_pos(self, coord, do_random_rotation=True):
        x, y, z = coord
        self._set_pos((x, y, z + self.z_offset))
        if do_random_rotation:
            self._set_random_hpr()

    def get_pos(self):
        x, y, z = self.pos_coords
        return x, y, z - self.z_offset

    def set_scale(self, scale):
        self.model.setScale(scale)


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
        self._set_collide_mask(BitMask32.bit(1))
        # self.model.flattenStrong()

    def set_pos(self, coord):
        x, y, z = coord
        self._set_pos((x, y, z))
