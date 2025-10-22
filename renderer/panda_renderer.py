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
    NodePath,
)

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

    # Board positioning
    BOARD_Y_OFFSET = -0.75  # Y-axis offset for board position (positive = away from camera)
    SUPPLY_Y_OFFSET = 6.0  # Y-axis offset for supply pool (positive = away from camera)

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
        self.rotation = math.pi / 6
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

        # Track world→board animations that need dynamic destination updates
        # Format: {marble_id: {'marble': marble, 'dst_local': (x,y,z), 'on_complete': callback}}
        self._world_to_board_animations = {}

        # Track board→world animations that need reparenting at animation start
        # Format: {marble_id: {'marble': marble, 'reparented': bool}}
        self._board_to_world_animations = {}

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

        # Create board container node for rotation (will be positioned at geometric center after board is built)
        self.board_node = NodePath("board_container")
        self.board_node.reparentTo(self.render)

        # anim: vx, vy, scale, skip

        self._build_base(board_layout)

        # Calculate geometric center and position board_node
        self._position_board_node()

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

        # Create player turn indicator
        self._create_player_indicator()

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
        # self.rotation += 0.002
        # if self.rotation > 2* math.pi:
        #     self.rotation = 0.
        self.set_board_rotation(self.rotation)

        # Handle board→world reparenting BEFORE animation updates
        self._handle_board_to_world_reparenting(task)

        # Update world→board animation destinations before animation update
        self._update_world_to_board_destinations()

        # Update state machine if active
        if (
            self.action_visualization_sequencer
            and self.action_visualization_sequencer.is_active()
        ):
            self.action_visualization_sequencer.update(task)

        # Delegate move/freeze animation updates to animation_manager
        completed = self.animation_manager.update(task)

        # Handle completed world→board animations
        self._handle_completed_world_to_board_animations(completed)

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

        x, y, z = 0, self.SUPPLY_Y_OFFSET, 0
        self._build_color_supply("b", self.black_marbles, y)
        y -= self.y_base_size
        self._build_color_supply("g", self.grey_marbles, y)
        y -= self.y_base_size
        self._build_color_supply("w", self.white_marbles, y)

    def _reposition_supply_marbles(self):
        """Reposition all supply marbles to their correct locations.

        Used after reset_board() to recalculate marble positions that may have
        been affected by board rotation. Supply marbles are in world space
        (not parented to board_node), so their positions don't auto-update.
        """
        y = self.SUPPLY_Y_OFFSET
        x_off = self.SUPPLY_MARBLE_SCALE * 2.0

        # Reposition black marbles
        supply_count = len(self.marble_supply["b"])
        xx = (self.x_base_size / 2.0 - (x_off * supply_count)) / 2.0
        for marble in self.marble_supply["b"]:
            marble.set_pos((xx, y, 0))
            marble.configure_as_supply_marble(self._supply_key(marble), self.SUPPLY_MARBLE_SCALE)
            xx += x_off

        # Reposition grey marbles
        y -= self.y_base_size
        supply_count = len(self.marble_supply["g"])
        xx = (self.x_base_size / 2.0 - (x_off * supply_count)) / 2.0
        for marble in self.marble_supply["g"]:
            marble.set_pos((xx, y, 0))
            marble.configure_as_supply_marble(self._supply_key(marble), self.SUPPLY_MARBLE_SCALE)
            xx += x_off

        # Reposition white marbles
        y -= self.y_base_size
        supply_count = len(self.marble_supply["w"])
        xx = (self.x_base_size / 2.0 - (x_off * supply_count)) / 2.0
        for marble in self.marble_supply["w"]:
            marble.set_pos((xx, y, 0))
            marble.configure_as_supply_marble(self._supply_key(marble), self.SUPPLY_MARBLE_SCALE)
            xx += x_off

    def _setup_camera(self):
        """Setup camera position and orientation based on board size."""
        # Get camera configuration for this board size
        config = self.CAMERA_CONFIG.get(self.rings, self.CAMERA_CONFIG[37])
        cam_dist = config["cam_dist"]
        cam_height = config["cam_height"]

        # After _position_board_node(), the board is centered at origin
        # Camera looks at board_node position (0, 0, 0)
        center_x, center_y, center_z = 0, 0, 0

        # Position camera behind (negative Y) and above (positive Z) the board
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

        # Calculate vectors from board center to corners and normalize to unit vectors
        tl_out = np.array(tl_coord) - self.geometric_center
        tl_out = tl_out / np.linalg.norm(tl_out)
        tr_out = np.array(tr_coord) - self.geometric_center
        tr_out = tr_out / np.linalg.norm(tr_out)
        # tl_coord += tl_out
        # tr_coord += tr_out

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
        p_ul = tl_coord + tl_out + d_ul

        # For player 2 (right side), we want the pool closer and further from camera
        # Use a smaller offset and add extra Y offset to move away from camera
        d_ur = (tr_coord - tr_adj) * self.capture_pool_offset_scale
        p_ur = tr_coord + tr_out + d_ur

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
                    # Pass board_node as parent so rings rotate with the board
                    base_piece = BasePiece(self, parent=self.board_node)
                    x = x_center + (k * self.x_base_size) - x_row_offset
                    y = y_center - y_row_offset
                    coords = (x, y, 0)
                    base_piece.set_pos(coords)
                    base_piece.configure_as_ring(pos)
                    self.pos_to_base[pos] = base_piece
                    self.pos_to_coords[pos] = coords

                    # Create coordinate label if show_coords is enabled
                    # Labels are also parented to board_node so they rotate with rings
                    if self.show_coords:
                        text_node = TextNode(f"label_{pos}")
                        text_node.setText(pos)
                        text_node.setAlign(TextNode.ACenter)
                        text_node.setTextColor(1, 1, 1, 1)  # White text
                        text_node_path = self.board_node.attachNewNode(text_node)
                        # Position label above the ring
                        label_z = 0.4  # Height above ring
                        text_node_path.setPos(x, y, label_z)
                        text_node_path.setScale(0.15)  # Text size
                        # Make text face camera (billboard effect)
                        text_node_path.setBillboardPointEye()
                        self.pos_to_label[pos] = text_node_path

            logger.debug(f"Board row {i}: {self.pos_array[i]}")

    def _position_board_node(self):
        """Calculate geometric center and offset all ring positions to center the board at origin.

        This ensures the board rotates around its geometric center while keeping board_node at origin.
        """
        if not self.pos_to_coords:
            logger.warning("No rings found, board_node positioned at origin")
            self.board_node.setPos(0, self.BOARD_Y_OFFSET, 0)
            self.geometric_center = np.array([0.0, 0.0, 0.0])
            return

        # Collect all ring coordinates (x, y, z tuples)
        all_coords = np.array([coords for coords in self.pos_to_coords.values()])

        # Calculate geometric center (mean of all ring positions)
        self.geometric_center = np.mean(all_coords, axis=0)

        logger.debug(f"Board geometric center: ({self.geometric_center[0]:.3f}, {self.geometric_center[1]:.3f}, {self.geometric_center[2]:.3f})")

        # Position board_node with configurable Y offset
        # X and Z remain at origin for clean rotation around geometric center
        self.board_node.setPos(0, self.BOARD_Y_OFFSET, 0)

        # Offset all rings by -geometric_center so they're centered around origin
        # This way, rotation around origin == rotation around geometric center
        for pos_str, base_piece in self.pos_to_base.items():
            world_coords = self.pos_to_coords[pos_str]
            # Calculate local position relative to board_node (which is at origin)
            local_coords = (
                world_coords[0] - self.geometric_center[0],
                world_coords[1] - self.geometric_center[1],
                world_coords[2] - self.geometric_center[2]
            )
            base_piece.set_pos(local_coords)
            # Update stored coordinates to be world coordinates (for camera, etc.)
            # Actually, keep them as original world coords for backward compatibility

        # Also offset coordinate labels if they exist
        for pos_str, label in self.pos_to_label.items():
            # Labels were already parented to board_node in _build_base
            # Just need to offset their positions
            world_coords = self.pos_to_coords[pos_str]
            label_z = 0.4  # Height above ring (same as in _build_base)
            local_coords = (
                world_coords[0] - self.geometric_center[0],
                world_coords[1] - self.geometric_center[1],
                world_coords[2] - self.geometric_center[2] + label_z
            )
            label.setPos(local_coords)

    def _world_to_board_local(self, world_coords):
        """Convert world coordinates to board-local coordinates.

        When board_node is rotated, marbles parented to it need positions in board-local space.
        This converts from stored world coordinates to the local coordinate system of board_node.

        Args:
            world_coords: Tuple of (x, y, z) in world space

        Returns:
            Tuple of (x, y, z) in board-local space
        """
        return (
            world_coords[0] - self.geometric_center[0],
            world_coords[1] - self.geometric_center[1],
            world_coords[2] - self.geometric_center[2]
        )

    def _board_local_to_world(self, local_coords):
        """Convert board-local coordinates to world coordinates.

        Takes board-local coordinates and returns the current world position,
        accounting for board rotation.

        Args:
            local_coords: Tuple of (x, y, z) in board-local space

        Returns:
            Tuple of (x, y, z) in world space
        """
        from panda3d.core import Point3

        local_point = Point3(local_coords[0], local_coords[1], local_coords[2])
        world_point = self.render.getRelativePoint(self.board_node, local_point)
        return (world_point.x, world_point.y, world_point.z)

    def _update_world_to_board_destinations(self):
        """Update animation destinations for world→board transitions as board rotates."""
        for marble_id, info in list(self._world_to_board_animations.items()):
            marble = info['marble']
            # Find the animation by matching entity reference
            for anim_item in self.animation_manager.current_animations:
                if anim_item.get('entity') is marble:
                    # Recalculate world position of destination
                    dst_local = info['dst_local']
                    dst_world = self._board_local_to_world(dst_local)
                    # Update the animation's destination
                    anim_item['dst'] = dst_world
                    break

    def _handle_completed_world_to_board_animations(self, completed_animations):
        """Handle completion of world→board animations - reparent and cleanup.

        Args:
            completed_animations: List of completed animation items from animation_manager
        """
        if not completed_animations:
            return

        for anim_item in completed_animations:
            entity = anim_item.get('entity')
            if entity is None:
                continue

            # Find this entity in our tracking dict by matching entity reference
            marble_id = id(entity)
            if marble_id in self._world_to_board_animations:
                info = self._world_to_board_animations.pop(marble_id)
                marble = info['marble']
                dst_local = info['dst_local']
                on_complete = info.get('on_complete')

                # Reparent to board_node and set final board-local position
                marble.model.reparentTo(self.board_node)
                marble.set_pos(dst_local)

                # Call additional completion callback if provided
                if on_complete:
                    on_complete()

    def _handle_board_to_world_reparenting(self, task):
        """Handle reparenting of board→world animations when they start.

        Marbles animating from board to capture pool stay parented to board_node during
        their defer period (rotating with the board). When the animation actually starts,
        we reparent them to render (world space) and update their source coordinates.

        Args:
            task: Current Panda3D task (for time information)
        """
        if not self._board_to_world_animations:
            return

        current_time = task.time

        for marble_id, info in list(self._board_to_world_animations.items()):
            if info['reparented']:
                continue  # Already handled

            marble = info['marble']

            # Find the animation for this marble
            for anim_item in self.animation_manager.current_animations:
                if anim_item.get('entity') is marble:
                    # Check if animation has started (passed its start_time)
                    start_time = anim_item.get('start_time', 0)
                    if current_time >= start_time:
                        # Animation is starting - do the reparenting now
                        # Get current board-local position
                        src_coords_local = marble.get_pos()
                        # Convert to world coordinates
                        src_coords_world = self._board_local_to_world(src_coords_local)

                        # Reparent to world space
                        marble.model.reparentTo(self.render)
                        # Set world position
                        marble.set_pos(src_coords_world)

                        # Update animation's source coordinates
                        anim_item['src'] = src_coords_world

                        # Mark as reparented
                        info['reparented'] = True
                    break

    def setup_lights(self):
        """Setup lighting with shadow-casting directional lights.

        Configures two directional lights with shadow mapping for better
        intra-object shadows (self-shadowing on curved surfaces like marbles).
        """

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
        a_light.setColor((0.12, 0.16, 0.16, 1.00))
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

            # Track this marble for reparenting when animation starts
            # For now, get placeholder source coordinates (will be updated at animation start)
            src_coords_local = captured_marble.get_pos()  # Current board-local position
            src_coords_world_placeholder = self._board_local_to_world(src_coords_local)

            # Queue animation - the src coordinates will be corrected when animation starts
            self.animation_manager.queue_animation(
                anim_type="move",
                entity=captured_marble,
                src=src_coords_world_placeholder,
                dst=capture_pool_coords,
                scale=self.CAPTURED_MARBLE_SCALE,
                duration=action_duration,
                defer=capture_animation_defer,
            )

            # Track this animation for reparenting at start time
            self._board_to_world_animations[id(captured_marble)] = {
                'marble': captured_marble,
                'reparented': False,
            }

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
        src_coords_world = put_marble.get_pos()  # Get world position (marble still in render)
        if put_marble not in [p for p, _ in mip]:
            mip.append((put_marble, src_coords_world))

        # Store current scale before configure (marble is at SUPPLY_MARBLE_SCALE or player pool scale)
        current_scale = put_marble.model.getScale()[0]  # Get current scale (uniform)

        # Configure marble's metadata (tags, collision masks, etc.) and position tracking
        self.pos_to_marble[dst] = put_marble
        put_marble.configure_as_board_marble(dst, self.BOARD_MARBLE_SCALE, f"board:{dst}")

        self._update_capture_marble_colliders()

        # Convert destination from world to board-local coordinates
        dst_coords_local = self._world_to_board_local(dst_coords)

        if action_duration == 0:
            # Instant placement: reparent and position immediately
            put_marble.model.reparentTo(self.board_node)
            put_marble.set_pos(dst_coords_local)
            put_marble.set_scale(self.BOARD_MARBLE_SCALE)
        else:
            # Animated placement: marble stays in world space, destination updates each frame
            # Calculate initial world position of destination
            dst_coords_world = self._board_local_to_world(dst_coords_local)

            # Restore original scale so animation can interpolate from current → target
            put_marble.set_scale(current_scale)

            # Queue animation using world coordinates
            # The animation manager will return an ID that we can track
            self.animation_manager.queue_animation(
                anim_type="move",
                entity=put_marble,
                src=src_coords_world,
                dst=dst_coords_world,
                scale=self.BOARD_MARBLE_SCALE,
                duration=action_duration,
                defer=delay,
            )

            # Track this animation for dynamic destination updates
            # We need to get the animation ID - let's read the queue to get the last queued item
            # Actually, we need to modify this approach - we can't get the ID until it's dequeued
            # Let me store a marker and match it when the animation starts
            # Better: store the marble reference and match by entity
            self._world_to_board_animations[id(put_marble)] = {
                'marble': put_marble,
                'dst_local': dst_coords_local,
            }

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

    def show_action(self, player, render_data, action_duration=0.0):
        """Visualize an action without highlights.

        Args:
            player: Player making the move
            render_data: RenderData value object containing action_dict
            action_duration: Animation duration
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

            # Convert world coordinates to board-local coordinates for marbles parented to board_node
            src_coords_local = self._world_to_board_local(src_coords)
            dst_coords_local = self._world_to_board_local(dst_coords)

            if action_duration == 0:
                action_marble.set_pos(dst_coords_local)
            else:
                self.animation_manager.queue_animation(
                    anim_type="move",
                    entity=action_marble,
                    src=src_coords_local,
                    dst=dst_coords_local,
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

        # 4. Return marbles to supply pools and clear their visual state
        for color, marbles in self.marbles_in_play.items():
            for marble, _ in marbles:  # Ignore stored position - will be recalculated
                self.marble_supply[color].append(marble)
                self._marble_registry[id(marble)] = marble
                marble.model.clearMaterial()  # Clear any highlight materials
                # Reparent to world space (marbles on board are parented to board_node)
                marble.model.reparentTo(self.render)

        # 5. Clear marbles_in_play dict (important - this accumulates otherwise!)
        self.marbles_in_play = self._make_marble_dict()

        # 6. Clear pos_to_marble dict (CRITICAL - stale entries prevent highlights!)
        self.pos_to_marble.clear()

        # 7. Rebuild player pools
        self._build_players_marble_supply()

        # 8. Reposition all supply marbles to correct locations
        # This recalculates positions accounting for any board rotation
        self._reposition_supply_marbles()

        # 9. Recreate highlight state machine
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
            self.show_action(player, render_data, task_delay_time)

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

    def _create_player_indicator(self) -> None:
        """Create on-screen text indicator showing which player's turn it is."""
        from direct.gui.OnscreenText import OnscreenText

        self.player_indicator = OnscreenText(
            text="",
            pos=(-1.3, -0.9),  # Bottom-left corner
            scale=0.065,
            fg=(1, 1, 1, 1),  # White text
            align=TextNode.ALeft,
            mayChange=True,
            font=self.loader.loadFont("cmtt12.egg"),  # Monospace font
            shadow=(0, 0, 0, 1),  # Black shadow for bold/readable effect
            shadowOffset=(0.04, 0.04),  # Shadow offset for depth
        )

    def update_player_indicator(self, player_number: int, notation: str = "") -> None:
        """Update the player indicator to show the current player and move notation.

        Args:
            player_number: The player number (1 or 2)
            notation: Optional move notation to display after the player number
        """
        if hasattr(self, 'player_indicator'):
            if notation:
                text = f"Player {player_number}: {notation}"
            else:
                text = f"Player {player_number}"
            self.player_indicator.setText(text)
            # Change color based on player
            if player_number == 1:
                self.player_indicator['fg'] = (0.0, 0.2, 1.0, 1)  # Dark blue for Player 1
            else:
                self.player_indicator['fg'] = (0.7, 0.2, 0.0, 1)  # Dark red for Player 2

    def set_board_rotation(self, angle_radians: float) -> None:
        """Set the rotation angle of the board around the Z-axis.

        The board rotates around an axis perpendicular to the board plane (Z-axis),
        passing through the geometric center. Only rings and marbles on rings rotate;
        supply and capture pools remain static.

        Args:
            angle_radians: Rotation angle in radians. Positive values rotate counter-clockwise
                          when viewed from above (looking down the Z-axis).

        Note:
            Board rotation is allowed during animations. Marbles parented to board_node
            use board-local coordinates, so they automatically rotate with the board without
            requiring position adjustment.
        """
        # Convert radians to degrees (Panda3D uses degrees for rotation)
        angle_degrees = math.degrees(angle_radians)

        # setH() sets the heading (rotation around Z-axis) in degrees
        self.board_node.setH(angle_degrees)

        logger.debug(f"Board rotation set to {angle_radians:.4f} radians ({angle_degrees:.2f} degrees)")
