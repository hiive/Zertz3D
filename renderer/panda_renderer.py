import logging
import math
import sys

from typing import Callable, Any, Optional

import numpy as np
import simplepbr
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    LVector4,
    BitMask32,
    DirectionalLight,
    WindowProperties,
    loadPrcFileData,
    TextNode,
)
from sympy import capture

from renderer.panda3d.action_sequencer import ActionVisualizationSequencer
from renderer.panda3d.animation_manager import AnimationManager
from renderer.panda3d.highlighting_manager import HighlightingManager
from renderer.panda3d.interaction_helper import InteractionHelper
from renderer.panda3d.material_modifier import MaterialModifier
from renderer.panda3d.water_node import WaterNode
from renderer.panda3d.models import BasePiece, SkyBox, make_marble
from shared.constants import MARBLE_TYPES, SUPPLY_CONTEXT_MAP
from shared.render_data import RenderData
from shared.materials_modifiers import (
    PLACEMENT_HIGHLIGHT_MATERIAL_MOD,
    REMOVABLE_HIGHLIGHT_MATERIAL_MOD,
    CAPTURE_HIGHLIGHT_MATERIAL_MOD,
    SELECTED_CAPTURE_MATERIAL_MOD,
    CAPTURE_FLASH_MATERIAL_MOD,
    HOVER_PRIMARY_MATERIAL_MOD,
    HOVER_SECONDARY_MATERIAL_MOD,
    HOVER_SUPPLY_MATERIAL_MOD,
    HOVER_CAPTURED_MATERIAL_MOD,
    SUPPLY_HIGHLIGHT_WHITE_MATERIAL_MOD,
    SUPPLY_HIGHLIGHT_GREY_MATERIAL_MOD,
    SUPPLY_HIGHLIGHT_BLACK_MATERIAL_MOD,
)


logger = logging.getLogger(__name__)


class PandaRenderer(ShowBase):
    # Rendering constants
    BASE_SIZE_X = 0.8
    BASE_SIZE_Y = 0.7

    BASE_SCALE_FACTOR = 0.8
    SUPPLY_MARBLE_SCALE = 0.275 * BASE_SCALE_FACTOR
    BOARD_MARBLE_SCALE = 0.35 * BASE_SCALE_FACTOR
    CAPTURED_MARBLE_SCALE = 0.9 * BOARD_MARBLE_SCALE
    CAPTURE_POOL_OFFSET_SCALE = 0.9 * BASE_SCALE_FACTOR
    CAPTURE_POOL_MEMBER_OFFSET_X = 0.9 * BASE_SCALE_FACTOR

    # Animation timing
    CAPTURE_FLASH_DURATION = 0.3  # Duration of yellow flash before capture

    # Shadow configuration
    SHADOW_MAP_RESOLUTION = 2048  # Resolution for shadow maps (higher = better quality, more VRAM)
    # SHADOW_BIAS: Adjust if shadow artifacts occur
    #   - Too low (< 0.001): "Shadow acne" (self-shadowing artifacts)
    #   - Too high (> 0.02): "Peter panning" (shadows detach from objects)
    #   - Default works well for most cases; uncomment and adjust if needed
    # SHADOW_BIAS = 0.005

    # Local copies keep renderer-specific tweaks isolated from shared constants.
    # todo check if these are read only usages. Can we freeze the source constants?
    MARBLE_ORDER = tuple(MARBLE_TYPES)
    SUPPLY_HIGHLIGHT_CONTEXTS = dict(SUPPLY_CONTEXT_MAP)
    CONTEXT_STYLES = {
        "placement": PLACEMENT_HIGHLIGHT_MATERIAL_MOD,
        "removal": REMOVABLE_HIGHLIGHT_MATERIAL_MOD,
        "capture_sources": CAPTURE_HIGHLIGHT_MATERIAL_MOD,
        "capture_destinations": SELECTED_CAPTURE_MATERIAL_MOD,
        # Supply marbles use green (placement color) for consistent visual language
        SUPPLY_HIGHLIGHT_CONTEXTS["w"]: PLACEMENT_HIGHLIGHT_MATERIAL_MOD, # SUPPLY_HIGHLIGHT_WHITE_MATERIAL_MOD
        SUPPLY_HIGHLIGHT_CONTEXTS["g"]: PLACEMENT_HIGHLIGHT_MATERIAL_MOD,
        SUPPLY_HIGHLIGHT_CONTEXTS["b"]: PLACEMENT_HIGHLIGHT_MATERIAL_MOD,
        "hover_primary": HOVER_PRIMARY_MATERIAL_MOD,
        "hover_secondary": HOVER_SECONDARY_MATERIAL_MOD,
        "hover_supply": HOVER_SUPPLY_MATERIAL_MOD,
        "hover_captured": HOVER_CAPTURED_MATERIAL_MOD,
    }

    # Camera configuration per board size
    CAMERA_CONFIG = {
        37: {"center_pos": "D4", "cam_dist": 10, "cam_height": 8},
        48: {"center_pos": "D5", "cam_dist": 10, "cam_height": 10},
        61: {"center_pos": "E5", "cam_dist": 11, "cam_height": 10},
    }

    def __init__(
        self,
        board_layout,
        white_marbles=6,
        grey_marbles=8,
        black_marbles=10,
        rings=37,
        show_coords=False,
        highlight_choices=False,
        update_callback=None,
        move_duration=0.666,
        start_delay=0.0,
    ):
        # Configure OpenGL version before initializing ShowBase
        loadPrcFileData("", "gl-version 3 2")
        super().__init__()

        self.highlight_choices = highlight_choices
        self.update_callback = update_callback
        self.move_duration = move_duration
        self.start_delay = start_delay

        props = WindowProperties()
        # props.setSize(1364, 768)
        # props.setSize(1024, 768)

        self.win.requestProperties(props)

        # Initialize animation manager for move and freeze animations
        self.animation_manager = AnimationManager(self)

        # Initialize highlighting manager for highlight animations
        self.highlighting_manager = HighlightingManager(self)

        self.pos_to_base = {}
        self.pos_to_marble = {}
        self._marble_registry: dict[int, Any] = {}
        self.removed_bases = []
        self.pos_to_coords = {}
        self.pos_to_label = {}  # Maps position strings to text labels
        self.pos_array = None
        self.show_coords = show_coords
        self._action_context = None  # Tracks in-flight action for completion callbacks

        self.x_base_size = self.BASE_SIZE_X
        self.y_base_size = self.BASE_SIZE_Y

        self.number_offset = (self.x_base_size / 2, self.y_base_size, 0)
        self.letter_offset = (self.x_base_size / 2, -self.y_base_size, 0)

        # Import board size constants
        from game.zertz_board import ZertzBoard

        # Configure letters based on board size (number of rings)
        if rings == ZertzBoard.SMALL_BOARD_37:
            self.letters = "ABCDEFG"
        elif rings == ZertzBoard.MEDIUM_BOARD_48:
            self.letters = "ABCDEFGH"
        elif rings == ZertzBoard.LARGE_BOARD_61:
            self.letters = "ABCDEFGHJ"
        else:
            raise ValueError(
                f"Unsupported board size: {rings} rings. "
                f"Supported sizes are {ZertzBoard.SMALL_BOARD_37}, {ZertzBoard.MEDIUM_BOARD_48}, and {ZertzBoard.LARGE_BOARD_61}."
            )

        self.capture_pool_offset_scale = self.CAPTURE_POOL_OFFSET_SCALE
        self.capture_pool_member_offset = (self.CAPTURE_POOL_MEMBER_OFFSET_X, 0, 0)

        self.capture_pools = None
        self.capture_pool_coords = None

        # Initialize interaction helper for mouse picking and hover detection
        self.interaction_helper = InteractionHelper(self)

        self.pipeline = simplepbr.init()
        self.pipeline.enable_shadows = True
        self.pipeline.use_330 = True
        # Increase shadow map resolution for better intra-object shadows (marbles, rings)
        self.pipeline.max_lights = 2  # Limit to 2 shadow-casting lights

        # Configure shadow quality
        # Note: Shadow bias can be adjusted if shadow acne or peter-panning occurs
        # Lower values (0.001-0.005): Less peter-panning, more shadow acne risk
        # Higher values (0.01-0.02): Less shadow acne, more peter-panning risk

        self.accept("escape", sys.exit)  # Escape quits
        self.accept("aspectRatioChanged", self._setup_water)
        self.disableMouse()  # Disable mouse camera control
        self.accept("mouse1", self.interaction_helper.on_mouse_click)

        # Store rings for later use in camera setup after board is built
        self.rings = rings

        # self.camera.setPosHpr(0, 0, 16, 0, 270, 0)  # Set the camera
        self.setup_lights()  # Setup default lighting

        self.sb = SkyBox(self)

        self.wb = None
        self._setup_water()

        # anim: vx, vy, scale, skip

        self._build_base(board_layout)

        self.marble_supply = None
        self.marbles_in_play = None
        self.white_marbles = white_marbles
        self.black_marbles = black_marbles
        self.grey_marbles = grey_marbles
        self._build_marble_supply()

        self._build_players_marble_supply()
        self._update_capture_marble_colliders()

        # Setup camera after board is built so we can center on it
        self._setup_camera()

        # Add renderer update task (sort=50, runs after game loop task at sort=49)
        self.task = self.taskMgr.add(self.update, "zertzUpdate", sort=50)

        # Setup game loop if callback was provided
        self.game_loop_task = None
        if self.update_callback is not None:
            self._setup_game_loop()

        # Initialize highlight state machine if highlight_choices is enabled
        self.action_visualization_sequencer = (
            ActionVisualizationSequencer(self) if self.highlight_choices else None
        )

    def attach_update_loop(
        self, update_fn: Callable[[], bool], interval: float
    ) -> bool:
        delay = max(interval, 0.0)

        def _task(task):
            if update_fn():
                return Task.done
            return Task.again

        self.taskMgr.doMethodLater(delay, _task, "zertzGameLoop")
        return True

    def _setup_game_loop(self):
        """Set up the game loop task with specified duration.

        Uses self.update_callback and self.move_duration set during initialization.
        """
        self.game_loop_task = self.taskMgr.doMethodLater(
            self.move_duration, self.update_callback, "update_game", sort=49
        )

    def update(self, task):
        """Update all animations - delegating to manager classes."""

        # Update state machine if active
        if (
            self.action_visualization_sequencer
            and self.action_visualization_sequencer.is_active()
        ):
            self.action_visualization_sequencer.update(task)

        # Delegate move/freeze animation updates to animation_manager
        self.animation_manager.update(task)

        # Delegate highlight animation updates to highlighting_manager
        self.highlighting_manager.update(task)

        self.interaction_helper.dispatch_hover_target()
        self._complete_action_if_ready()
        return task.cont

    def get_current_pos(self, anim_factor, dst, src, jump=True):
        sx, sy, sz = src
        dx, dy, dz = dst
        dsx = dx - sx
        dsy = dy - sy
        x = sx + dsx * anim_factor
        y = sy + dsy * anim_factor

        xy_dist = np.sqrt(dsx * dsx + dsy * dsy)
        zc_scale = 1.25
        zc = (
            0
            if (not jump or xy_dist == 0)
            else np.log(xy_dist) * np.sin(anim_factor * math.pi) * zc_scale
        )
        z = sz + (dz - sz) * anim_factor
        return x, y, z + zc

    def _build_color_supply(self, color, supply_count, y, z=0):
        x_off = self.SUPPLY_MARBLE_SCALE * 2.0
        xx = (self.x_base_size / 2.0 - (x_off * supply_count)) / 2.0
        for k in range(supply_count):
            mb = make_marble(self, color)
            mb.set_pos((xx, y, z))
            mb.configure_as_supply_marble(self._supply_key(mb), self.SUPPLY_MARBLE_SCALE)
            self.marble_supply[color].append(mb)
            self._marble_registry[id(mb)] = mb
            xx += x_off

    def _make_marble_dict(self):
        return {"w": [], "b": [], "g": []}

    def _build_marble_supply(self):
        if self.marble_supply is not None:
            for _, marbles in self.marble_supply.items():
                for marble in marbles:
                    self._marble_registry.pop(id(marble), None)
                    marble.removeNode()
        self.marble_supply = self._make_marble_dict()
        self.marbles_in_play = self._make_marble_dict()

        x, y, z = 0, 5.25, 0
        self._build_color_supply("b", self.black_marbles, y)
        y -= self.y_base_size
        self._build_color_supply("g", self.grey_marbles, y)
        y -= self.y_base_size
        self._build_color_supply("w", self.white_marbles, y)

    def _setup_camera(self):
        """Setup camera position and orientation based on board size."""
        # Get camera configuration for this board size
        config = self.CAMERA_CONFIG.get(self.rings, self.CAMERA_CONFIG[37])
        center_pos = config["center_pos"]
        cam_dist = config["cam_dist"]
        cam_height = config["cam_height"]

        # Get the actual 3D coordinates of the center position
        if center_pos in self.pos_to_coords:
            center_x, center_y, center_z = self.pos_to_coords[center_pos]
        else:
            # Fallback to origin if we can't find the center
            center_x, center_y, center_z = 0, 0, 0

        # Position camera to look at the board center
        # Camera is positioned behind (negative Y) and above (positive Z) the center point
        cam_x = center_x
        cam_y = center_y - cam_dist
        cam_z = center_z + cam_height

        self.camera.setPos(cam_x, cam_y, cam_z)
        self.camera.lookAt(center_x, center_y, center_z)

    def _setup_water(self):
        if self.wb is not None:
            self.wb.remove()
            self.wb.destroy()

        # distort: offset, strength, refraction factor (0 = perfect mirror, 1 = total refraction), refractivity
        # l_texcoord1.xy = vtx_texcoord0.xy * k_wateranim.z + k_wateranim.xy * k_time.x
        # Make water plane larger to cover the entire visible area
        self.wb = WaterNode(
            self,
            -15,
            -10,
            15,
            10,
            0,
            # anim: vx, vy, scale, skip
            anim=LVector4(0.0245, -0.0122, 1.5, 1),
            distort=LVector4(0.2, 0.05, 0.8, 0.2),
        )  # (0, 0, .5, 0))

    def _build_players_marble_supply(self):
        self.capture_pools = {1: self._make_marble_dict(), 2: self._make_marble_dict()}
        self.capture_pool_coords = {1: [], 2: []}

        # Find corner positions dynamically based on board size
        # Top-left corner: first letter, highest number in that column
        # Top-right corner: last letter, highest number in that column
        first_letter = self.letters[0]
        last_letter = self.letters[-1]

        # Find the highest row for each corner
        # [pos for pos in self.pos_to_coords.keys() if pos.startswith(first_letter)]
        # [pos for pos in self.pos_to_coords.keys() if pos.startswith(last_letter)]

        # Get positions with max number (top of board)
        # Rightmost non-empty value in the first row
        first_row = self.pos_array[0]
        top_right = first_row[first_row != ""][-1]

        # First value in the first row
        top_left = first_row[0]

        logger.debug(f"Board: top_left={top_left}, top_right={top_right}")
        # Get adjacent positions to calculate board direction vectors
        # Find a neighbor position to determine the board's edge direction
        # second_letter = self.letters[1]
        # second_from_top_left = [pos for pos in self.pos_to_coords.keys()
        #                         if pos.startswith(second_letter) and int(pos[1:]) == int(top_left[1:])]

        # second_last_letter = self.letters[-2]
        # second_from_top_right = [pos for pos in self.pos_to_coords.keys()
        #                          if pos.startswith(second_last_letter) and int(pos[1:]) == int(top_right[1:])]

        # Rightmost non-empty value in the second row
        second_row = self.pos_array[1]
        second_from_top_right = second_row[second_row != ""][-2]

        # first value in second row
        second_from_top_left = second_row[1]

        # logger.debug(f"Board: second_from_top_left={second_from_top_left}, second_from_top_right={second_from_top_right}")
        # Get coordinates
        tl_coord = self.pos_to_coords[top_left]
        tr_coord = self.pos_to_coords[top_right]

        # Use adjacent positions if they exist, otherwise use same row one position down
        if second_from_top_left:
            tl_adj = self.pos_to_coords[second_from_top_left]
        else:
            # Fallback: use position one row down in same column
            adj_pos = f"{first_letter}{int(top_left[1:]) - 1}"
            tl_adj = self.pos_to_coords.get(adj_pos, tl_coord)

        if second_from_top_right:
            tr_adj = self.pos_to_coords[second_from_top_right]
        else:
            adj_pos = f"{last_letter}{int(top_right[1:]) - 1}"
            tr_adj = self.pos_to_coords.get(adj_pos, tr_coord)

        # Calculate direction vectors from corner to adjacent position using numpy arrays
        tl_coord = np.array(tl_coord)
        tl_adj = np.array(tl_adj)
        tr_coord = np.array(tr_coord)
        tr_adj = np.array(tr_adj)
        capture_pool_member_offset = np.array(self.capture_pool_member_offset)

        d_ul = (tl_coord - tl_adj) * self.capture_pool_offset_scale
        p_ul = tl_coord + d_ul

        # For player 2 (right side), we want the pool closer and further from camera
        # Use a smaller offset and add extra Y offset to move away from camera
        d_ur = (tr_coord - tr_adj) * self.capture_pool_offset_scale
        p_ur = tr_coord + d_ur

        # Create 12 positions per player (6 + 6 with offset)
        for r in range(2):
            for i in range(6):
                pp1 = (
                    p_ul + (-d_ur * (i / 1.5)) + (-capture_pool_member_offset * (r + 1))
                )
                pp1[2] += 0.25
                self.capture_pool_coords[1].append(tuple(pp1))

                pp2 = p_ur + (-d_ul * (i / 1.5)) + (capture_pool_member_offset * (r + 1))
                pp2[2] += 0.25
                self.capture_pool_coords[2].append(tuple(pp2))

        self._update_capture_marble_colliders()

    def _init_pos_coords(self):
        self.pos_to_base.clear()
        self.pos_to_coords.clear()
        self.pos_array = None

    def _build_base(self, board_layout):
        self._init_pos_coords()
        self.pos_array = board_layout

        # Calculate 3D positions for rendering
        r_max = len(self.letters)
        is_even = r_max % 2 == 0
        h_max = lambda xx: r_max - abs(
            self.letters.index(self.letters[xx]) - (r_max // 2)
        )
        r_min = h_max(0)
        if is_even:
            r_min += 1
        x_center = -(self.x_base_size / 2) * r_max / 2
        y_center = (self.y_base_size / 2) * r_max / 2

        # Create 3D base pieces for each position in the layout
        for i in range(r_max):
            hh = h_max(i)
            ll = self.letters[:hh] if i < hh / 2 else self.letters[-hh:]
            x_row_offset = self.x_base_size / 2 * (h_max(i) - r_min)
            y_row_offset = self.y_base_size * i

            for k in range(len(ll)):
                lt = ll[k]
                pa = self.letters.find(lt)
                pos = self.pos_array[i][pa]

                if pos != "":  # Only create pieces for non-empty positions
                    base_piece = BasePiece(self)
                    x = x_center + (k * self.x_base_size) - x_row_offset
                    y = y_center - y_row_offset
                    coords = (x, y, 0)
                    base_piece.set_pos(coords)
                    base_piece.configure_as_ring(pos)
                    self.pos_to_base[pos] = base_piece
                    self.pos_to_coords[pos] = coords

                    # Create coordinate label if show_coords is enabled
                    if self.show_coords:
                        text_node = TextNode(f"label_{pos}")
                        text_node.setText(pos)
                        text_node.setAlign(TextNode.ACenter)
                        text_node.setTextColor(1, 1, 1, 1)  # White text
                        text_node_path = self.render.attachNewNode(text_node)
                        # Position label above the ring
                        label_z = 0.4  # Height above ring
                        text_node_path.setPos(x, y, label_z)
                        text_node_path.setScale(0.15)  # Text size
                        # Make text face camera (billboard effect)
                        text_node_path.setBillboardPointEye()
                        self.pos_to_label[pos] = text_node_path

            logger.debug(f"Board row {i}: {self.pos_array[i]}")

    def setup_lights(self):
        """Setup lighting with shadow-casting directional lights.

        Configures two directional lights with shadow mapping for better
        intra-object shadows (self-shadowing on curved surfaces like marbles).
        """
        from panda3d.core import OrthographicLens

        # Main directional light (primary shadow caster)
        p_light = DirectionalLight("p_light")
        p_node = self.render.attachNewNode(p_light)
        p_node.setPos(-12, -2, 12)
        p_node.lookAt(0, 0, 0)
        p_light.setColor((1, 1, 1, 1))
        # Enable shadow casting with configurable resolution
        p_light.setShadowCaster(True, self.SHADOW_MAP_RESOLUTION, self.SHADOW_MAP_RESOLUTION)

        # Configure shadow camera lens to cover the board area
        # Use orthographic lens for directional light shadows
        # Film size needs to cover largest board (61 rings) plus player pools and marble supplies
        lens = p_light.getLens()
        lens.setFilmSize(40, 40)  # Large enough for all board sizes (37, 48, 61 rings)
        lens.setNearFar(1, 50)    # Near/far planes for shadow depth range

        self.render.setLight(p_node)
        p_node.hide(BitMask32(1))

        # Secondary directional light (also casts shadows)
        s_light1 = DirectionalLight("s_light1")
        s_light1.setColor((0.75, 0.75, 0.75, 1))
        s_node1 = self.render.attachNewNode(s_light1)
        s_node1.setPos(0, 0, 20)
        s_node1.lookAt(0, 0, 0)
        # Enable shadow casting for fill light
        s_light1.setShadowCaster(True, self.SHADOW_MAP_RESOLUTION, self.SHADOW_MAP_RESOLUTION)

        # Configure shadow camera lens for secondary light
        lens1 = s_light1.getLens()
        lens1.setFilmSize(40, 40)  # Match main light coverage
        lens1.setNearFar(5, 50)

        s_node1.hide(BitMask32(1))
        self.render.setLight(s_node1)

        # Ambient light (no shadows, provides base illumination)
        # Higher ambient light makes shadows more subtle/softer
        a_light = AmbientLight("a_light")
        a_light.setColor((0.06, 0.08, 0.10, 1.00))
        a_node = self.render.attachNewNode(a_light)
        a_node.hide(BitMask32(1))
        self.render.setLight(a_node)

    def _animate_marble_to_capture_pool(
        self, captured_marble, src_pos_str, src_coords, player, marble_color, action_duration
    ):
        """Animate a captured marble moving to player's capture pool.

        Args:
            captured_marble: The marble entity to animate
            src_pos_str: Source position string (e.g., 'D4') for flash highlight
            src_coords: Source coordinates (where the marble is coming from)
            player: Player capturing the marble
            marble_color: Color of the captured marble ('w', 'g', or 'b')
            action_duration: Animation duration (0 for instant positioning)
        """
        # Add to player's captured marbles
        captured_marbles = self.capture_pools[player.n][marble_color]
        captured_count = sum(
            [len(k) for k in self.capture_pools[player.n].values()]
        )
        captured_marbles.append(captured_marble)

        # Store current scale before configure (marble is at BOARD_MARBLE_SCALE)
        current_scale = captured_marble.model.getScale()[0]  # Get current scale (uniform)

        # Configure marble's metadata (tags, collision masks, etc.) which also sets target scale
        captured_marble.configure_as_captured_marble(
            player.n,
            self._captured_key(captured_marble),
            self.CAPTURED_MARBLE_SCALE
        )
        self._update_capture_marble_colliders()

        # Clamp captured count if needed
        if captured_count >= len(self.capture_pool_coords[player.n]):
            if captured_count > len(self.capture_pool_coords[player.n]):
                logger.error(
                    f"Captured marbles count ({captured_count}) exceeds available coords ({len(self.capture_pool_coords[player.n])}) for player {player.n}"
                )
            captured_count = len(self.capture_pool_coords[player.n]) - 1

        capture_pool_coords = self.capture_pool_coords[player.n][captured_count]

        # Either instantly position or animate to capture pool
        if action_duration == 0:
            captured_marble.set_pos(capture_pool_coords)
            captured_marble.set_scale(self.CAPTURED_MARBLE_SCALE)
        else:
            # Restore original scale so animation can interpolate from current → target
            captured_marble.set_scale(current_scale)

            # Wait for capturing marble to complete its jump before moving captured marble
            capture_animation_defer = action_duration

            # Apply yellow flash only if highlight_choices is enabled
            if self.highlight_choices:
                # Use the marble's captured key to reference it in the highlighting system
                marble_key = self._captured_key(captured_marble)

                # Queue the flash highlight to start when capturing marble lands
                self.queue_highlight(
                    [marble_key],
                    self.CAPTURE_FLASH_DURATION,
                    material_mod=CAPTURE_FLASH_MATERIAL_MOD,
                    defer=action_duration,
                )

                # Defer the capture animation so it starts after the flash
                capture_animation_defer = action_duration + self.CAPTURE_FLASH_DURATION

            # Queue animation which will interpolate scale from current to CAPTURED_MARBLE_SCALE
            self.animation_manager.queue_animation(
                anim_type="move",
                entity=captured_marble,
                src=src_coords,
                dst=capture_pool_coords,
                scale=self.CAPTURED_MARBLE_SCALE,
                duration=action_duration,
                defer=capture_animation_defer,
            )

    def show_isolated_removal(self, player, pos, marble_color, action_duration=0):
        """Animate removal of an isolated ring (with or without marble).

        Args:
            action_duration: Animation duration (already scaled by controller)
        """
        # Remove the ring base piece
        if pos in self.pos_to_base:
            base_piece = self.pos_to_base[pos]
            base_pos = base_piece.get_pos()

            if action_duration == 0:
                base_piece.hide()
            else:
                self.animation_manager.queue_animation(
                    anim_type="move",
                    entity=base_piece,
                    src=base_pos,
                    dst=None,
                    scale=None,
                    duration=action_duration,
                    defer=action_duration,
                )
            self.removed_bases.append((base_piece, base_pos))

        # If there's a marble, remove it and add to player's captured pool
        if marble_color is not None and pos in self.pos_to_marble:
            captured_marble = self.pos_to_marble.pop(pos)
            src_coords = captured_marble.get_pos()

            self._animate_marble_to_capture_pool(
                captured_marble, pos, src_coords, player, marble_color, action_duration
            )

    def show_marble_placement(self, player, action_dict, action_duration=0., delay=0.):
        """Place a marble on the board (PUT action only, no ring removal).

        Args:
            action_dict: Dictionary of actions to place.
            action_duration: Animation duration
            delay: Delay in seconds
        """
        action_marble_color = action_dict["marble"]
        dst = action_dict["dst"]
        dst_coords = self.pos_to_coords[dst]

        # add marble from supply
        supply = self.marble_supply[action_marble_color]
        if len(supply) == 0:
            supply = self.capture_pools[player.n][action_marble_color]
        if len(supply) == 0:
            logger.error(
                f"No marbles available in supply for player {player.n}, color {action_marble_color}"
            )
            return
        put_marble = supply.pop()
        mip = self.marbles_in_play[action_marble_color]
        src_coords = put_marble.get_pos()
        if put_marble not in [p for p, _ in mip]:
            mip.append((put_marble, src_coords))

        # Store current scale before configure (marble is at SUPPLY_MARBLE_SCALE or player pool scale)
        current_scale = put_marble.model.getScale()[0]  # Get current scale (uniform)

        # Configure marble's metadata (tags, collision masks, etc.) and position tracking
        self.pos_to_marble[dst] = put_marble
        put_marble.configure_as_board_marble(dst, self.BOARD_MARBLE_SCALE, f"board:{dst}")
        self._update_capture_marble_colliders()

        if action_duration == 0:
            put_marble.set_pos(dst_coords)
            put_marble.set_scale(self.BOARD_MARBLE_SCALE)
        else:
            # Restore original scale so animation can interpolate from current → target
            put_marble.set_scale(current_scale)

            # Queue animation which will interpolate scale from current to BOARD_MARBLE_SCALE
            self.animation_manager.queue_animation(
                anim_type="move",
                entity=put_marble,
                src=src_coords,
                dst=dst_coords,
                scale=self.BOARD_MARBLE_SCALE,
                duration=action_duration,
                defer=delay,
            )

    def show_ring_removal(self, action_dict, action_duration=0., delay=0.):
        """Remove a ring from the board (PUT action only).

        Args:
            action_dict: Action dictionary with 'remove' key
            action_duration: Animation duration (already scaled by controller)
            delay: Delay (in seconds)
        """
        base_piece_id = action_dict["remove"]
        if base_piece_id != "":
            if base_piece_id in self.pos_to_base:
                base_piece = self.pos_to_base[base_piece_id]
                base_pos = base_piece.get_pos()
                if action_duration == 0:
                    base_piece.hide()
                else:
                    # Queue ring removal animation
                    removal_defer = action_duration + delay
                    self.animation_manager.queue_animation(
                        anim_type="move",
                        entity=base_piece,
                        src=base_pos,
                        dst=None,
                        scale=None,
                        duration=action_duration,
                        defer=removal_defer,
                    )

                self.removed_bases.append((base_piece, base_pos))

    def show_action(self, player, render_data, action_duration=0.0, action_result=None):
        """Visualize an action without highlights.

        Args:
            player: Player making the move
            render_data: RenderData value object containing action_dict
            action_duration: Animation duration
            action_result: ActionResult containing frozen positions (optional)
        """
        # Extract data from value objects
        action_dict = render_data.action_dict

        # Player 2: {'action': 'PUT', 'marble': 'g',              'dst': 'G2', 'remove': 'D0'}
        # Player 1: {'action': 'CAP', 'marble': 'g', 'src': 'G2', 'dst': 'E2', 'capture': 'b'}
        # Player 1: {'action': 'PASS'}
        action = action_dict["action"]

        # PASS actions have no visual component
        if action == "PASS":
            return

        # action_dict['marble']
        dst = action_dict["dst"]
        dst_coords = self.pos_to_coords[dst]

        if action == "PUT":
            # Call the split methods (duration already scaled by controller)
            self.show_marble_placement(player, action_dict, action_duration, self.start_delay)
            self.show_ring_removal(action_dict, action_duration, self.start_delay)
            self.start_delay = 0
        elif action == "CAP":
            src = action_dict["src"]
            src_coords = self.pos_to_coords[src]
            cap = action_dict["cap"]
            cap_coords = self.pos_to_coords[cap]
            captured_marble_color = action_dict["capture"]
            action_marble = self.pos_to_marble.pop(src)
            captured_marble = self.pos_to_marble.pop(cap)
            self.pos_to_marble[dst] = action_marble
            action_marble.configure_as_board_marble(dst, self.BOARD_MARBLE_SCALE, f"board:{dst}")
            if action_duration == 0:
                action_marble.set_pos(dst_coords)
            else:
                self.animation_manager.queue_animation(
                    anim_type="move",
                    entity=action_marble,
                    src=src_coords,
                    dst=dst_coords,
                    scale=self.BOARD_MARBLE_SCALE,
                    duration=action_duration,
                    defer=0,
                )

            # Animate captured marble to player's pool
            self._animate_marble_to_capture_pool(
                captured_marble,
                cap,
                cap_coords,
                player,
                captured_marble_color,
                action_duration,
            )

    def reset_board(self):
        # 1. Clear animations FIRST (before resetting visuals)
        # Clear move/freeze animations in animation_manager
        self.animation_manager.clear()

        # Clear highlight animations in highlighting_manager
        self.highlighting_manager.clear()

        self.clear_context_highlights()

        # 2. Restore removed rings and make them visible again
        for b, pos in self.removed_bases:
            b.set_pos(pos)
            b.show()  # Make visible again
        self.removed_bases.clear()

        # 3. Clear ALL ring visual state (materials, transparency, color scale)
        for pos_str, base_piece in self.pos_to_base.items():
            base_piece.model.clearMaterial()
            base_piece.model.clearColorScale()
            base_piece.model.clearTransparency()

        # 4. Move marbles back to pool and clear their visual state
        for color, marbles in self.marbles_in_play.items():
            for marble, pos in marbles:

                self.marble_supply[color].append(marble)
                self._marble_registry[id(marble)] = marble

                marble.model.clearMaterial()  # Clear any highlight materials
                marble.set_pos(pos)
                marble.configure_as_supply_marble(self._supply_key(marble), self.SUPPLY_MARBLE_SCALE)

        # 5. Clear marbles_in_play dict (important - this accumulates otherwise!)
        self.marbles_in_play = self._make_marble_dict()

        # 6. Clear pos_to_marble dict (CRITICAL - stale entries prevent highlights!)
        self.pos_to_marble.clear()

        # 7. Rebuild player pools
        self._build_players_marble_supply()

        # 8. Recreate highlight state machine
        if self.highlight_choices:
            self.action_visualization_sequencer = ActionVisualizationSequencer(self)
        self._action_context = None

    def is_busy(self):
        """Check if renderer is busy with highlights or animations.

        Returns:
            True if state machine is active or animations are running
        """
        if (
            self.action_visualization_sequencer
            and self.action_visualization_sequencer.is_active()
        ):
            return True
        return self.is_animation_active()

    def execute_action(
        self,
        player: Any,
        render_data: RenderData,
        action_result: Any,
        task_delay_time: float,
        on_complete: Callable[[Any, Any], None] | None,
    ) -> None:
        """Execute an action with optional highlighting.

        Args:
            player: Player making the move
            render_data: RenderData value object containing action_dict and optional highlight data
            action_result: ActionResult from game.take_action() (encapsulates captures and frozen positions)
            task_delay_time: Task delay time for animations
            on_complete: Callback invoked when all animations/highlights finish
        """
        self._set_action_context(
            player, render_data, action_result, task_delay_time, on_complete
        )

        if self.highlight_choices and self.action_visualization_sequencer:
            # Start highlighting - pass render_data and action_result to state machine
            self.action_visualization_sequencer.start(
                player, render_data, task_delay_time
            )
        else:
            # Direct visualization without highlights
            self.show_action(player, render_data, task_delay_time, action_result)

        # If no animations or highlights were queued, complete immediately
        self._complete_action_if_ready()

    def queue_highlight(
        self,
        rings,
        duration,
        material_mod=None,
        defer=0,
    ):
        """Add a highlight to the queue.

        Args:
            rings: List of position strings (e.g., ['A1', 'B2', 'C3'])
            duration: How long to show highlight in seconds
            material_mod: Material object (defaults to PLACEMENT_HIGHLIGHT_MATERIAL)
            defer: Delay before starting (seconds)
        """
        # Delegate to highlighting_manager
        self.highlighting_manager.queue_highlight(rings, duration, material_mod, defer)

    def is_animation_active(self):
        """Check if any animations (move/freeze/highlight) are active or queued."""
        # Check both animation_manager (move/freeze) and highlighting_manager (highlight) animations
        return (
            self.animation_manager.is_active()
            or self.highlighting_manager.is_active()
        )

    def set_selection_callback(
        self, callback: Optional[Callable[[dict], None]]
    ) -> None:
        """Set callback for mouse click selection events.

        Args:
            callback: Function to call with selection dict when an entity is clicked
        """
        self.interaction_helper.set_selection_callback(callback)

    def set_hover_callback(
        self, callback: Optional[Callable[[dict | None], None]]
    ) -> None:
        """Set callback for hover events.

        Args:
            callback: Function to call with hover dict (or None when hover ends)
        """
        self.interaction_helper.set_hover_callback(callback)

    def show_hover_feedback(
        self,
        primary: Optional[set[str] | list[str]] = None,
        secondary: Optional[set[str] | list[str]] = None,
        supply_colors: Optional[set[str] | list[str]] = None,
        captured_targets: Optional[set[tuple[int, str]] | list[tuple[int, str]]] = None,
    ) -> None:
        """Display visual feedback for hovering over entities.

        Args:
            primary: Primary positions to highlight
            secondary: Secondary positions to highlight
            supply_colors: Supply marble colors to highlight
            captured_targets: Captured marble targets to highlight (owner, color)
        """
        self.interaction_helper.show_hover_feedback(
            primary, secondary, supply_colors, captured_targets
        )

    def clear_hover_highlights(self) -> None:
        """Clear all hover-related highlights."""
        self.interaction_helper.clear_hover_highlights()

    @staticmethod
    def _supply_key(marble: Any) -> str:
        try:
            key = marble.getPythonTag("zertz_key")  # type: ignore[attr-defined]
            if key:
                return str(key)
        except Exception:
            pass
        return f"supply:{id(marble)}"

    def _supply_highlight_keys(self, color: str) -> list[str]:
        marbles = self.marble_supply.get(color, []) if self.marble_supply else []
        return [self._supply_key(marble) for marble in marbles if marble is not None]

    @staticmethod
    def _captured_key(marble: Any) -> str:
        try:
            key = marble.getPythonTag("zertz_key")  # type: ignore[attr-defined]
            if key:
                return str(key)
        except Exception:
            pass
        return f"captured:{id(marble)}"

    def _captured_highlight_keys(self, owner: int, color: str) -> list[str]:
        if not self.capture_pools or owner not in self.capture_pools:
            return []
        marbles = self.capture_pools[owner].get(color, [])
        return [self._captured_key(marble) for marble in marbles if marble is not None]

    def _clear_supply_highlights(self) -> None:
        for context in self.SUPPLY_HIGHLIGHT_CONTEXTS.values():
            self.clear_highlight_context(context)

    def _apply_supply_highlights(self, valid_color_indices: set[int]) -> None:
        for idx, color in enumerate(self.MARBLE_ORDER):
            context = self.SUPPLY_HIGHLIGHT_CONTEXTS[color]
            if idx in valid_color_indices:
                positions = self._supply_highlight_keys(color)
                if positions:
                    self.highlight_context(context, positions)
                    continue
            self.clear_highlight_context(context)

    def _is_supply_empty(self) -> bool:
        if not self.marble_supply:
            return True
        return all(len(marbles) == 0 for marbles in self.marble_supply.values())

    def _update_capture_marble_colliders(self) -> None:
        enabled = self._is_supply_empty()
        mask = BitMask32.bit(1) if enabled else BitMask32.allOff()
        if not self.capture_pools:
            return
        for pool in self.capture_pools.values():
            for marbles in pool.values():
                for marble in marbles:
                    marble.model.setCollideMask(mask)

    def apply_context_masks(self, board, placement_mask, capture_mask) -> None:
        """Translate action masks into highlight sets for the current board."""
        width = board.width

        if placement_mask is None or capture_mask is None:
            self.clear_highlight_context()
            self._clear_supply_highlights()
            return

        if np.any(capture_mask):
            capture_sources: set[str] = set()
            capture_destinations: set[str] = set()
            for direction in range(capture_mask.shape[0]):
                ys, xs = np.where(capture_mask[direction])
                for y, x in zip(ys, xs):
                    label_src = board.index_to_str((y, x))
                    if label_src:
                        capture_sources.add(label_src)
                    dy, dx = board.DIRECTIONS[direction]
                    cap_index = (y + dy, x + dx)
                    dst_index = board.get_jump_destination((y, x), cap_index)
                    if dst_index is None:
                        continue
                    dst_y, dst_x = dst_index
                    if (
                        0 <= dst_y < width
                        and 0 <= dst_x < width
                        and board.state[board.RING_LAYER, dst_y, dst_x] == 1
                    ):
                        label_dst = board.index_to_str(dst_index)
                        if label_dst:
                            capture_destinations.add(label_dst)

            if capture_sources:
                self.highlight_context("capture_sources", capture_sources)
            else:
                self.clear_highlight_context("capture_sources")

            if capture_destinations:
                self.highlight_context("capture_destinations", capture_destinations)
            else:
                self.clear_highlight_context("capture_destinations")

            self.clear_highlight_context("placement")
            self.clear_highlight_context("removal")
            self._clear_supply_highlights()
            return

        placement_rings: set[str] = set()
        removal_rings: set[str] = set()
        valid_colors: set[int] = set()

        for marble_idx in range(placement_mask.shape[0]):
            put_indices, rem_indices = np.where(placement_mask[marble_idx])
            if put_indices.size > 0:
                valid_colors.add(marble_idx)
            for put, rem in zip(put_indices, rem_indices):
                put_y, put_x = divmod(put, width)
                label_put = board.index_to_str((put_y, put_x))
                if label_put:
                    placement_rings.add(label_put)
                if rem != width**2:
                    rem_y, rem_x = divmod(rem, width)
                    label_rem = board.index_to_str((rem_y, rem_x))
                    if label_rem:
                        removal_rings.add(label_rem)

        # Show only placement rings (not removal rings) in apply_context_masks
        # Removal ring highlighting is handled separately by the hover feedback system
        # which shows them only after a marble has been placed (proper phasing)
        if placement_rings:
            self.highlight_context("placement", placement_rings)
        else:
            self.clear_highlight_context("placement")

        # Don't highlight removal rings here - let hover feedback handle phasing
        self.clear_highlight_context("removal")

        self._apply_supply_highlights(valid_colors)
        self.clear_highlight_context("capture_sources")
        self.clear_highlight_context("capture_destinations")

    def set_context_highlights(
        self,
        context: str,
        positions: list[str] | set[str],
        material_mod: MaterialModifier | None = None,
    ) -> None:
        """Apply persistent highlights for a named context."""
        resolved_material_mod = self._resolve_context_style(context, material_mod)
        # Delegate to highlighting_manager
        self.highlighting_manager.set_context_highlights(context, positions, resolved_material_mod)

    def clear_context_highlights(self, context: str | None = None) -> None:
        """Clear context highlights for a specific context or all contexts."""
        # Delegate to highlighting_manager
        self.highlighting_manager.clear_context_highlights(context)

    def highlight_context(self, context: str, positions: set[str] | list[str]) -> None:
        material_mod = self._resolve_context_style(context, None)
        self.set_context_highlights(context, positions, material_mod)

    def clear_highlight_context(self, context: str | None = None) -> None:
        self.clear_context_highlights(context)

    def _resolve_context_style(
        self,
        context: str,
        material_mod: MaterialModifier | None,
    ) -> MaterialModifier:
        if material_mod is not None:
            return material_mod

        # Get default material/style from context
        default_mod = self.CONTEXT_STYLES.get(
            context,
            PLACEMENT_HIGHLIGHT_MATERIAL_MOD,  # Default to Material object
        )
        return material_mod or default_mod

    @property
    def current_action_result(self):
        """Return the in-flight action result (if any)."""
        if self._action_context:
            return self._action_context["action_result"]
        return None

    def _set_action_context(
        self,
        player: Any,
        render_data: RenderData,
        action_result: Any,
        task_delay_time: float,
        on_complete: Callable[[Any, Any], None] | None,
    ) -> None:
        """Store state for the current action until animations complete."""
        if self._action_context is not None:
            logger.warning(
                "Renderer action context replaced before completion; forcing completion."
            )
            self._complete_action()

        self._action_context = {
            "player": player,
            "render_data": render_data,
            "action_result": action_result,
            "task_delay_time": task_delay_time,
            "on_complete": on_complete,
        }

    def _complete_action_if_ready(self) -> None:
        """Invoke completion callback when highlights and animations finish."""
        if self._action_context is None:
            return
        if (
            self.action_visualization_sequencer
            and self.action_visualization_sequencer.is_active()
        ):
            return
        if self.is_animation_active():
            return
        self._complete_action()

    def _complete_action(self) -> None:
        """Call the completion callback (if any) and clear context."""
        if self._action_context is None:
            return

        context = self._action_context
        self._action_context = None

        callback = context.get("on_complete")
        if callback:
            callback(context["player"], context["action_result"])

    def report_status(self, message: str) -> None:
        """Handle textual status reports for compatibility with composite renderers."""
        logger.info(message)
