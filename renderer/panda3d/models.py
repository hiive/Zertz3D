from abc import ABC
import random

from panda3d.core import TextureAttrib, AmbientLight, BitMask32


class _BaseModel(ABC):
    def __init__(self, renderer, model_name, color=(1.0, 1.0, 1.0, 1.0), parent=None):
        self.renderer = renderer
        self.model_name = model_name
        self.color = color
        self.pos_coords = None
        self.saved_coords = None
        self.model = None
        self.parent = parent  # Optional parent NodePath for scene graph organization
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
        # Use custom parent if provided, otherwise default to renderer.render
        parent_node = self.parent if self.parent is not None else self.renderer.render
        self.model.reparentTo(parent_node)
        self._set_collide_mask(BitMask32.allOff())

    def show(self):
        if self.model is not None:
            self.model.show()

    def hide(self):
        if self.model is not None:
            self.model.hide()

    def set_collide_mask(self, mask):
        """Set collision mask for this model (public wrapper for _set_collide_mask)."""
        self._set_collide_mask(mask)

    def set_python_tag(self, tag_name, value):
        """Set a Python tag on the model."""
        if self.model is not None:
            self.model.setPythonTag(tag_name, value)

    def clear_python_tag(self, tag_name):
        """Clear a Python tag from the model."""
        if self.model is not None:
            self.model.clearPythonTag(tag_name)

    def clear_visual_effects(self):
        """Clear all visual effects (material, color scale, transparency)."""
        if self.model is not None:
            self.model.clearMaterial()
            self.model.clearColorScale()
            self.model.clearTransparency()

    def set_color_scale(self, r, g, b, alpha):
        """Set color scale for the model."""
        if self.model is not None:
            self.model.setColorScale(r, g, b, alpha)

    def set_transparency(self, mode):
        """Set transparency mode for the model."""
        if self.model is not None:
            self.model.setTransparency(mode)


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


def make_marble(renderer, color, parent=None):
    if color == "w":
        return WhiteBallModel(renderer, color, parent)
    elif color == "b":
        return BlackBallModel(renderer, color, parent)
    elif color == "g":
        return GrayBallModel(renderer, color, parent)
    return None


class _BallBase(_BaseModel):
    def __init__(self, renderer, rgba_color, marble_color, parent=None):
        model_name = "models/ball_lo.bam"
        super().__init__(renderer, model_name, rgba_color, parent)
        # self.model.flattenStrong()
        # Store marble color as readonly property (set once at creation)
        self.zertz_color = marble_color
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

    def configure_as_supply_marble(self, key, scale):
        """Configure this marble as a supply marble (available to both players).

        Args:
            key: Unique identifier for this marble in the supply
            scale: Scale factor for the marble
        """
        self.set_scale(scale)
        self.set_python_tag("zertz_entity", "supply_marble")
        self.set_python_tag("zertz_color", self.zertz_color)
        self.set_python_tag("zertz_key", key)
        self.set_collide_mask(BitMask32.bit(1))

    def configure_as_board_marble(self, label, scale, key=None):
        """Configure this marble as a board marble (placed on the board).

        Args:
            label: Position label (e.g., 'A1', 'B2')
            scale: Scale factor for the marble
            key: Optional unique identifier (defaults to 'board:{label}')
        """
        if key is None:
            key = f"board:{label}"
        self.set_scale(scale)
        self.set_python_tag("zertz_entity", "board_marble")
        self.set_python_tag("zertz_label", label)
        self.set_python_tag("zertz_color", self.zertz_color)
        self.set_python_tag("zertz_key", key)
        self.set_collide_mask(BitMask32.bit(1))

    def configure_as_captured_marble(self, owner, key, scale):
        """Configure this marble as a captured marble (owned by a player).

        Args:
            owner: Player number (1 or 2)
            key: Unique identifier for this captured marble
            scale: Scale factor for the marble
        """
        self.set_scale(scale)
        self.set_python_tag("zertz_entity", "captured_marble")
        self.clear_python_tag("zertz_label")
        self.set_python_tag("zertz_color", self.zertz_color)
        self.set_python_tag("zertz_owner", owner)
        self.set_python_tag("zertz_key", key)
        self.set_collide_mask(BitMask32.allOff())


class BlackBallModel(_BallBase):
    def __init__(self, renderer, marble_color, parent=None):
        rgba_color = (0.25, 0.25, 0.25, 1)
        super().__init__(renderer, rgba_color, marble_color, parent)


class GrayBallModel(_BallBase):
    def __init__(self, renderer, marble_color, parent=None):
        rgba_color = (1.25, 1.25, 1.25, 1)
        super().__init__(renderer, rgba_color, marble_color, parent)


class WhiteBallModel(_BallBase):
    def __init__(self, renderer, marble_color, parent=None):
        rgba_color = (1.5, 1.5, 1.2, 1)
        super().__init__(renderer, rgba_color, marble_color, parent)
        for c in self.model.findAllMatches("**/+GeomNode"):
            gn = c.node()
            for i in range(gn.getNumGeoms()):
                state = gn.getGeomState(i)
                state = state.removeAttrib(TextureAttrib.getClassType())
                gn.setGeomState(i, state)

    def _set_random_hpr(self):
        pass


class BasePiece(_BaseModel):
    def __init__(self, renderer, parent=None):
        color = (0.0, 0.0, 0.0, 1)
        model_name = "models/torus_lo.bam"

        super().__init__(renderer, model_name, color, parent)
        self.model.setP(self.model, 180)
        self._set_collide_mask(BitMask32.bit(1))
        # self.model.flattenStrong()

    def set_pos(self, coord):
        x, y, z = coord
        self._set_pos((x, y, z))

    def configure_as_ring(self, label):
        """Configure this base piece as a board ring.

        Args:
            label: Position label (e.g., 'A1', 'B2')
        """
        self.set_python_tag("zertz_entity", "ring")
        self.set_python_tag("zertz_label", label)
