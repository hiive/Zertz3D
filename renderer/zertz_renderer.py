import logging
import math
import sys

from queue import SimpleQueue, Empty
from typing import Callable, Any

import numpy as np
import simplepbr
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from panda3d.core import AmbientLight, LVector4, BitMask32, DirectionalLight, WindowProperties, loadPrcFileData, Material, TransparencyAttrib, TextNode

from renderer.water_node import WaterNode
from renderer.zertz_models import BasePiece, SkyBox, make_marble
from shared.render_data import RenderData

logger = logging.getLogger(__name__)


class ActionVisualizationSequencer:
    """Manages the multiphase highlighting sequence for showing moves."""

    # Phase constants
    PHASE_PLACEMENT_HIGHLIGHTS = 'placement_highlights'
    PHASE_SELECTED_PLACEMENT = 'selected_placement'
    PHASE_REMOVAL_HIGHLIGHTS = 'removal_highlights'
    PHASE_SELECTED_REMOVAL = 'selected_removal'
    PHASE_CAPTURE_HIGHLIGHTS = 'capture_highlights'
    PHASE_SELECTED_CAPTURE = 'selected_capture'
    PHASE_ANIMATING = 'animating'  # Waiting for final move animations to complete

    # Highlight and animation durations (seconds)
    ISOLATION_HIGHLIGHT_DURATION = 0.5  # Yellow flash for newly frozen rings
    FREEZE_FADE_DURATION = 0.3          # Fade to alpha 0.7 for frozen rings

    def __init__(self, renderer):
        """Initialize the state machine.

        Args:
            renderer: ZertzRenderer instance
        """
        self.renderer = renderer
        self.highlight_durations = {
            self.PHASE_PLACEMENT_HIGHLIGHTS: 0.5,
            self.PHASE_SELECTED_PLACEMENT: 0.5,
            self.PHASE_REMOVAL_HIGHLIGHTS: 0.5,
            self.PHASE_SELECTED_REMOVAL: 0.5,
            self.PHASE_CAPTURE_HIGHLIGHTS: 0.5,
            self.PHASE_SELECTED_CAPTURE: 0.5,
        }

        # State tracking
        self.phase = None  # Current phase: 'placement_highlights', 'selected_placement', etc.
        self.pending_player = None  # Player making the move
        self.pending_action_dict = None  # Action dict
        self.placement_positions = None  # List of position strings for placement highlights
        self.capture_moves = None  # List of capture move dicts
        self.removal_positions = None  # List of removable position strings
        self.task_delay_time = 0  # Animation duration from controller's task

    def is_active(self):
        """Check if the state machine is currently active."""
        return self.phase is not None

    def start(self, player, render_data, task_delay_time):
        """Start the highlighting sequence for an action.

        Args:
            player: Player making the move
            render_data: RenderData value object with action_dict and highlight data
            task_delay_time: Animation duration from controller's task
        """
        self.pending_player = player
        self.pending_action_dict = render_data.action_dict
        self.task_delay_time = task_delay_time
        self.placement_positions = render_data.placement_positions
        self.capture_moves = render_data.capture_moves
        self.removal_positions = render_data.removal_positions

        action_type = render_data.action_dict['action']
        if action_type == "PUT":
            # Queue placement highlights and start the sequence
            self._queue_placement_highlights(render_data.placement_positions)
            self.phase = self.PHASE_PLACEMENT_HIGHLIGHTS
        elif action_type == "CAP":
            # Queue capture highlights and start the sequence
            self._queue_capture_highlights(render_data.capture_moves)
            self.phase = self.PHASE_CAPTURE_HIGHLIGHTS
        elif action_type == "PASS":
            # PASS has no visuals, action already executed by controller
            self.phase = None  # Done immediately (no highlight phases)

    def update(self, task):
        """Update the state machine. Called each frame.

        Args:
            task: Panda3D task object

        Returns:
            True if the state machine should continue (waiting for animations or processing)
            False if the state machine is done and game should proceed
        """
        if not self.is_active():
            return False

        # Check if animations are still active
        if self.renderer.is_animation_active():
            return True  # Still waiting for animations to finish

        # Animations finished, advance to next phase
        if self.phase == self.PHASE_PLACEMENT_HIGHLIGHTS:
            self._on_placement_highlights_done()
        elif self.phase == self.PHASE_SELECTED_PLACEMENT:
            self._on_selected_placement_done(task)
        elif self.phase == self.PHASE_REMOVAL_HIGHLIGHTS:
            self._on_removal_highlights_done()
        elif self.phase == self.PHASE_SELECTED_REMOVAL:
            self._on_selected_removal_done(task)
        elif self.phase == self.PHASE_CAPTURE_HIGHLIGHTS:
            self._on_capture_highlights_done()
        elif self.phase == self.PHASE_SELECTED_CAPTURE:
            self._on_selected_capture_done(task)
        elif self.phase == self.PHASE_ANIMATING:
            # Waiting for final move animations to complete
            # When we reach here, animations are done, so phase can be set to None
            self.phase = None

        return self.is_active()

    def _queue_placement_highlights(self, placement_positions):
        """Queue highlights for all valid placement positions.

        Args:
            placement_positions: List of position strings (pre-converted by controller)
        """
        if placement_positions:
            self.renderer.queue_highlight(placement_positions, self.highlight_durations[self.PHASE_PLACEMENT_HIGHLIGHTS])

    def _queue_removal_highlights(self, removal_positions, defer=0):
        """Queue highlights for all valid removal positions for this action.

        Args:
            removal_positions: List of position strings (pre-converted by controller)
            defer: Delay before starting the highlight (seconds)
        """
        if removal_positions:
            self.renderer.queue_highlight(
                removal_positions,
                self.highlight_durations[self.PHASE_REMOVAL_HIGHLIGHTS],
                color=self.renderer.REMOVABLE_HIGHLIGHT_COLOR,
                emission=self.renderer.REMOVABLE_HIGHLIGHT_EMISSION,
                defer=defer
            )

    def _queue_capture_highlights(self, capture_moves):
        """Queue highlights for all valid capture moves, grouped by source marble.

        If only one capture is available, skip highlighting (will auto-advance to selected_capture phase).

        Args:
            capture_moves: List of capture move dicts (pre-converted by controller)
        """
        # Skip highlighting if only one capture available
        if capture_moves and len(capture_moves) == 1:
            print(f"[StateMachine] Skipping capture highlights - only 1 capture available")
            return

        # Group captures by source position
        captures_by_source = {}  # {src_str: [dst_str1, dst_str2, ...]}
        for action_dict in capture_moves:
            src_str = action_dict['src']
            dst_str = action_dict['dst']

            if src_str not in captures_by_source:
                captures_by_source[src_str] = set()
            if dst_str:
                captures_by_source[src_str].add(dst_str)

        print(f"[StateMachine] Queueing capture highlights: {len(capture_moves)} total captures, {len(captures_by_source)} source groups")

        # Queue highlights sequentially - each group displays one after another
        capture_duration = self.highlight_durations[self.PHASE_CAPTURE_HIGHLIGHTS]
        defer_time = 0
        for src_str, destinations in captures_by_source.items():
            # Highlight the source marble and all its possible destinations
            highlight_rings = [src_str] + list(destinations)
            print(f"[StateMachine]   Group: src={src_str}, destinations={list(destinations)}, defer={defer_time}")
            self.renderer.queue_highlight(
                highlight_rings,
                capture_duration,
                color=self.renderer.CAPTURE_HIGHLIGHT_COLOR,
                emission=self.renderer.CAPTURE_HIGHLIGHT_EMISSION,
                defer=defer_time
            )
            # Next group starts when this one ends
            defer_time += capture_duration

    def _on_placement_highlights_done(self):
        """Handle completion of placement highlights phase."""
        action_dict = self.pending_action_dict

        # Queue highlight for selected placement ring only
        selected_ring = action_dict['dst']
        self.renderer.queue_highlight([selected_ring], self.highlight_durations[self.PHASE_SELECTED_PLACEMENT])
        self.phase = self.PHASE_SELECTED_PLACEMENT

    def _on_selected_placement_done(self, task):
        """Handle completion of selected placement highlight phase."""
        player = self.pending_player
        action_dict = self.pending_action_dict

        # Animate marble placement (action already executed by controller)
        self.renderer.show_marble_placement(player, action_dict, self.task_delay_time)

        # Queue removal highlights AFTER marble placement animation completes
        # task_delay_time already contains the animation duration from controller
        if self.removal_positions:
            self._queue_removal_highlights(self.removal_positions, defer=self.task_delay_time)

        # Move to removal highlights phase
        self.phase = self.PHASE_REMOVAL_HIGHLIGHTS

    def _on_removal_highlights_done(self):
        """Handle completion of removal highlights phase."""
        action_dict = self.pending_action_dict

        # Queue highlight for selected removal ring only
        selected_removal = action_dict['remove']
        if selected_removal:  # Only if a ring is being removed
            self.renderer.queue_highlight(
                [selected_removal],
                self.highlight_durations[self.PHASE_SELECTED_REMOVAL],
                color=self.renderer.REMOVABLE_HIGHLIGHT_COLOR,
                emission=self.renderer.REMOVABLE_HIGHLIGHT_EMISSION
            )
        self.phase = self.PHASE_SELECTED_REMOVAL

    def _on_selected_removal_done(self, task):
        """Handle completion of selected removal highlight phase."""
        action_dict = self.pending_action_dict

        # Now animate ring removal only (marble was already placed)
        self.renderer.show_ring_removal(action_dict, self.task_delay_time)

        # Queue isolation highlight and freeze animation for newly frozen rings
        action_result = self.renderer.current_action_result
        newly_frozen = action_result.newly_frozen_positions if action_result else None
        if newly_frozen and self.renderer.show_moves:
            # Calculate timing: isolation highlight starts when ring removal completes
            removal_defer = self.task_delay_time

            # 1. Flash yellow highlight
            self.renderer.queue_highlight(
                list(newly_frozen),
                self.ISOLATION_HIGHLIGHT_DURATION,
                color=self.renderer.ISOLATION_HIGHLIGHT_COLOR,
                emission=self.renderer.ISOLATION_HIGHLIGHT_EMISSION,
                defer=removal_defer
            )

            # 2. Fade to alpha 0.7 (starts after yellow flash)
            freeze_defer = removal_defer + self.ISOLATION_HIGHLIGHT_DURATION
            self.renderer.animation_queue.put({
                'type': 'freeze',
                'positions': list(newly_frozen),
                'duration': self.FREEZE_FADE_DURATION,
                'defer': freeze_defer
            })

        # Wait for final move animations to complete
        self.phase = self.PHASE_ANIMATING

    def _on_capture_highlights_done(self):
        """Handle completion of capture highlights phase."""
        action_dict = self.pending_action_dict

        # Queue highlight for selected capture (src and dst) only in cornflower blue
        src_ring = action_dict['src']
        dst_ring = action_dict['dst']
        selected_rings = [src_ring, dst_ring]

        self.renderer.queue_highlight(
            selected_rings,
            self.highlight_durations[self.PHASE_SELECTED_CAPTURE],
            color=self.renderer.SELECTED_CAPTURE_HIGHLIGHT_COLOR,
            emission=self.renderer.SELECTED_CAPTURE_HIGHLIGHT_EMISSION
        )
        self.phase = self.PHASE_SELECTED_CAPTURE

    def _on_selected_capture_done(self, task):
        """Handle completion of selected capture highlight phase."""
        player = self.pending_player
        action_dict = self.pending_action_dict

        # Animate capture action (action already executed by controller)
        # Create minimal RenderData with just the action_dict
        render_data = RenderData(action_dict)
        self.renderer.show_action(player, render_data, self.task_delay_time, self.renderer.current_action_result)

        # Wait for final move animations to complete
        self.phase = self.PHASE_ANIMATING



class ZertzRenderer(ShowBase):
    # Rendering constants
    BASE_SIZE_X = 0.8
    BASE_SIZE_Y = 0.7
    POOL_MARBLE_SCALE = 0.25
    BOARD_MARBLE_SCALE = 0.35
    CAPTURED_MARBLE_SCALE_FACTOR = 0.9
    PLAYER_POOL_OFFSET_SCALE = 0.8
    PLAYER_POOL_MEMBER_OFFSET_X = 0.6

    # Move visualization
    PLACEMENT_HIGHLIGHT_COLOR = LVector4(0.0, 0.4, 0.0, 1)      # Dark green base
    PLACEMENT_HIGHLIGHT_EMISSION = LVector4(0.0, 0.08, 0.0, 1)  # Subtle green glow
    REMOVABLE_HIGHLIGHT_COLOR = LVector4(0.4, 0.0, 0.0, 1)      # Dark red base
    REMOVABLE_HIGHLIGHT_EMISSION = LVector4(0.08, 0.0, 0.0, 1)  # Subtle red glow
    CAPTURE_HIGHLIGHT_COLOR = LVector4(0.0, 0.0, 0.4, 1)        # Dark blue base
    CAPTURE_HIGHLIGHT_EMISSION = LVector4(0.0, 0.0, 0.08, 1)    # Subtle blue glow
    SELECTED_CAPTURE_HIGHLIGHT_COLOR = LVector4(0.39, 0.58, 0.93, 1)      # Cornflower blue
    SELECTED_CAPTURE_HIGHLIGHT_EMISSION = LVector4(0.08, 0.12, 0.19, 1)   # Cornflower blue glow
    ISOLATION_HIGHLIGHT_COLOR = LVector4(0.8, 0.8, 0.0, 1)      # Bright yellow
    ISOLATION_HIGHLIGHT_EMISSION = LVector4(0.16, 0.16, 0.0, 1)  # Yellow glow

    # Camera configuration per board size
    CAMERA_CONFIG = {
        37: {'center_pos': 'D4', 'cam_dist': 10, 'cam_height': 8},
        48: {'center_pos': 'D5', 'cam_dist': 10, 'cam_height': 10},
        61: {'center_pos': 'E5', 'cam_dist': 11, 'cam_height': 10}
    }

    def __init__(self, board_layout, white_marbles=6, grey_marbles=8, black_marbles=10, rings=37, show_coords=False, show_moves=False, update_callback=None, move_duration=0.666):
        # Configure OpenGL version before initializing ShowBase
        loadPrcFileData("", "gl-version 3 2")
        super().__init__()

        self.show_moves = show_moves
        self.update_callback = update_callback
        self.move_duration = move_duration

        props = WindowProperties()
        # props.setSize(1364, 768)
        # props.setSize(1024, 768)

        self.win.requestProperties(props)

        self.animation_queue = SimpleQueue()
        self.current_animations = []  # List of active animation items (both moves and highlights)
        self.pos_to_base = {}
        self.pos_to_marble = {}
        self.removed_bases = []
        self.pos_to_coords = {}
        self.pos_to_label = {}  # Maps position strings to text labels
        self.pos_array = None
        self.show_coords = show_coords
        self._animation_id_counter = 0  # For generating unique IDs
        self._action_context = None  # Tracks in-flight action for completion callbacks

        self.x_base_size = self.BASE_SIZE_X
        self.y_base_size = self.BASE_SIZE_Y
        self.pool_marble_scale = self.POOL_MARBLE_SCALE
        self.board_marble_scale = self.BOARD_MARBLE_SCALE
        self.captured_marble_scale = self.CAPTURED_MARBLE_SCALE_FACTOR * self.board_marble_scale

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

        self.player_pool_offset_scale = self.PLAYER_POOL_OFFSET_SCALE
        self.player_pool_member_offset = (self.PLAYER_POOL_MEMBER_OFFSET_X, 0, 0)

        self.player_pools = None
        self.player_pool_coords = None

        self.pipeline = simplepbr.init()
        self.pipeline.enable_shadows = True
        self.pipeline.use_330 = True

        self.accept('escape', sys.exit)  # Escape quits
        self.accept('aspectRatioChanged', self._setup_water)
        self.disableMouse()  # Disable mouse camera control

        # Store rings for later use in camera setup after board is built
        self.rings = rings

        # self.camera.setPosHpr(0, 0, 16, 0, 270, 0)  # Set the camera
        self.setup_lights()  # Setup default lighting

        self.sb = SkyBox(self)

        self.wb = None
        self._setup_water()

        # anim: vx, vy, scale, skip

        self._build_base(board_layout)

        self.marble_pool = None
        self.marbles_in_play = None
        self.white_marbles = white_marbles
        self.black_marbles = black_marbles
        self.grey_marbles = grey_marbles
        self._build_marble_pool()

        self._build_players_marble_pool()

        # Setup camera after board is built so we can center on it
        self._setup_camera()

        # Add renderer update task (sort=50, runs after game loop task at sort=49)
        self.task = self.taskMgr.add(self.update, 'zertzUpdate', sort=50)

        # Setup game loop if callback was provided
        self.game_loop_task = None
        if self.update_callback is not None:
            self._setup_game_loop()

        # Initialize highlight state machine if show_moves is enabled
        self.action_visualization_sequencer = ActionVisualizationSequencer(self) if self.show_moves else None

    def attach_update_loop(self, update_fn: Callable[[], bool], interval: float) -> bool:
        delay = max(interval, 0.0)

        def _task(task):
            if update_fn():
                return Task.done
            return Task.again

        self.taskMgr.doMethodLater(delay, _task, 'zertzGameLoop')
        return True

    def _setup_game_loop(self):
        """Setup the game loop task with specified duration.

        Uses self.update_callback and self.move_duration set during initialization.
        """
        self.game_loop_task = self.taskMgr.doMethodLater(
            self.move_duration,
            self.update_callback,
            'update_game',
            sort=49
        )

    def update(self, task):
        """Update all animations - both move and highlight types."""

        # Update state machine if active
        if self.action_visualization_sequencer and self.action_visualization_sequencer.is_active():
            self.action_visualization_sequencer.update(task)

        # Process animation queue - add new animations
        while not self.animation_queue.empty():
            try:
                anim_item = self.animation_queue.get_nowait()
                anim_type = anim_item.get('type', 'move')  # Default to 'move' for backward compatibility

                # Generate unique ID
                self._animation_id_counter += 1
                anim_item['id'] = self._animation_id_counter

                # Set timing
                defer = anim_item.get('defer', 0)
                anim_item['insert_time'] = task.time
                anim_item['start_time'] = task.time + defer
                anim_item['end_time'] = anim_item['start_time'] + anim_item['duration']

                if anim_type == 'move':
                    # For move animations, store initial scale
                    entity = anim_item['entity']
                    if 'src_scale' not in anim_item:
                        anim_item['src_scale'] = entity.get_scale()
                    anim_item['dst_scale'] = anim_item.get('scale')
                elif anim_type == 'highlight':
                    # For highlights, apply immediately (materials don't animate)
                    if task.time >= anim_item['start_time']:
                        self._apply_highlight(anim_item)

                self.current_animations.append(anim_item)
            except Empty:
                pass

        # Update all active animations
        to_remove = []
        for anim_item in self.current_animations:
            anim_type = anim_item.get('type', 'move')

            # For highlights, apply when start time is reached (if not already applied)
            if anim_type == 'highlight' and 'original_materials' not in anim_item:
                if task.time >= anim_item['start_time']:
                    self._apply_highlight(anim_item)

            # Check if animation hasn't started yet
            if task.time < anim_item['start_time']:
                continue

            # Check if animation has ended
            if task.time >= anim_item['end_time']:
                if anim_type == 'move':
                    # Set final position and scale
                    entity = anim_item['entity']
                    dst_scale = anim_item.get('dst_scale')
                    dst = anim_item.get('dst')
                    if dst_scale is not None:
                        entity.set_scale(dst_scale)
                    if dst is not None:
                        entity.set_pos(dst)
                    else:
                        # dst=None means this is a removal animation - hide the entity
                        entity.hide()
                elif anim_type == 'highlight':
                    # Clear highlight
                    self._clear_highlight(anim_item)
                elif anim_type == 'freeze':
                    # Ensure final alpha is set
                    positions = anim_item.get('positions', [])
                    target_alpha = 0.7
                    for pos_str in positions:
                        if pos_str in self.pos_to_base:
                            base_piece = self.pos_to_base[pos_str]
                            base_piece.model.setColorScale(1, 1, 1, target_alpha)
                            base_piece.model.setTransparency(TransparencyAttrib.MAlpha)

                to_remove.append(anim_item)
                continue

            # Update animation
            if anim_type == 'move':
                entity = anim_item['entity']
                src = anim_item['src']
                dst = anim_item.get('dst')
                src_scale = anim_item.get('src_scale')
                dst_scale = anim_item.get('dst_scale')
                duration = anim_item['duration']
                start_time = anim_item['start_time']
                elapsed_time = task.time - start_time
                anim_factor = elapsed_time / duration

                # Update scale
                if src_scale is not None and dst_scale is not None:
                    sx = src_scale.x + (dst_scale - src_scale.x) * anim_factor
                    entity.set_scale(sx)

                # Update position
                jump = True
                if dst is None:
                    # Disappearing animation (ring removal)
                    dx, dy, dz = src
                    adx = -1 if dx < 0 else 1
                    ady = -1 if dy < 0 else 1
                    dx = max(10, abs(dx) * 10) * adx
                    dy = max(4, abs(dy) * 4) * ady
                    fz = 1.5
                    dst = (dx, dy, dz*fz)
                    jump = False

                x, y, z = self.get_current_pos(anim_factor, dst, src, jump=jump)
                entity.set_pos((x, y, z))
            elif anim_type == 'highlight':
                # Update highlight blend (pulsing effect)
                original_materials = anim_item.get('original_materials', {})
                target_color = anim_item.get('target_color', self.PLACEMENT_HIGHLIGHT_COLOR)
                target_emission = anim_item.get('target_emission', self.PLACEMENT_HIGHLIGHT_EMISSION)
                duration = anim_item['duration']
                start_time = anim_item['start_time']
                elapsed_time = task.time - start_time

                # Use sine wave for smooth pulsing: fade in (0 to 1), fade out (1 to 0)
                # sin goes from 0 -> 1 -> 0 over the animation duration
                pulse_factor = math.sin((elapsed_time / duration) * math.pi)

                # Update material for each entity in the highlight
                for pos_str, mat_info in original_materials.items():
                    (original_mat, entity_type, original_color, original_emission,
                     original_metallic, original_roughness) = mat_info

                    # Get the entity
                    entity = None
                    if entity_type == 'marble' and pos_str in self.pos_to_marble:
                        entity = self.pos_to_marble[pos_str]
                    elif entity_type == 'ring' and pos_str in self.pos_to_base:
                        entity = self.pos_to_base[pos_str]

                    if entity is not None:
                        # Blend between original and target colors
                        blended_color = LVector4(
                            original_color.x + (target_color.x - original_color.x) * pulse_factor,
                            original_color.y + (target_color.y - original_color.y) * pulse_factor,
                            original_color.z + (target_color.z - original_color.z) * pulse_factor,
                            original_color.w + (target_color.w - original_color.w) * pulse_factor
                        )
                        blended_emission = LVector4(
                            original_emission.x + (target_emission.x - original_emission.x) * pulse_factor,
                            original_emission.y + (target_emission.y - original_emission.y) * pulse_factor,
                            original_emission.z + (target_emission.z - original_emission.z) * pulse_factor,
                            original_emission.w + (target_emission.w - original_emission.w) * pulse_factor
                        )

                        # Create and apply blended material
                        blended_mat = Material()
                        blended_mat.setMetallic(original_metallic)
                        blended_mat.setRoughness(original_roughness)
                        blended_mat.setBaseColor(blended_color)
                        blended_mat.setEmission(blended_emission)
                        entity.model.setMaterial(blended_mat, 1)
            elif anim_type == 'freeze':
                # Update freeze animation (alpha fade from 1.0 to 0.7)
                positions = anim_item.get('positions', [])
                target_alpha = 0.7
                start_alpha = 1.0
                duration = anim_item['duration']
                start_time = anim_item['start_time']
                elapsed_time = task.time - start_time
                anim_factor = elapsed_time / duration

                # Linear interpolation from 1.0 to 0.7
                current_alpha = start_alpha + (target_alpha - start_alpha) * anim_factor

                # Apply alpha to all frozen rings
                for pos_str in positions:
                    if pos_str in self.pos_to_base:
                        base_piece = self.pos_to_base[pos_str]
                        base_piece.model.setColorScale(1, 1, 1, current_alpha)
                        base_piece.model.setTransparency(TransparencyAttrib.MAlpha)

        # Remove completed animations
        for anim_item in to_remove:
            self.current_animations.remove(anim_item)

        self._complete_action_if_ready()
        return task.cont

    def get_current_pos(self, anim_factor, dst, src, jump=True):
        sx, sy, sz = src
        dx, dy, dz = dst
        dsx = (dx - sx)
        dsy = (dy - sy)
        x = sx + dsx * anim_factor
        y = sy + dsy * anim_factor

        xy_dist = np.sqrt(dsx*dsx + dsy*dsy)
        zc_scale = 1.25
        zc = 0 if (jump == False or xy_dist == 0) else np.log(xy_dist) * np.sin(anim_factor * math.pi) * zc_scale
        z = sz + (dz - sz) * anim_factor
        return x, y, z + zc

    def _build_color_pool(self, color, pool_count, y, z=0):
        x_off = self.pool_marble_scale * 2.0
        xx = (self.x_base_size / 2.0 - (x_off * pool_count)) / 2.0
        for k in range(pool_count):
            mb = make_marble(self, color)
            mb.set_scale(self.pool_marble_scale)
            mb.set_pos((xx, y, z))
            self.marble_pool[color].append(mb)
            xx += x_off

    def _make_marble_dict(self):
        return {'w': [], 'b': [], 'g': []}

    def _build_marble_pool(self):
        if self.marble_pool is not None:
            for _, marbles in self.marble_pool.items():
                for marble in marbles:
                    marble.removeNode()
        self.marble_pool = self._make_marble_dict()
        self.marbles_in_play = self._make_marble_dict()

        x, y, z = 0, 5.25, 0
        self._build_color_pool('b', self.black_marbles, y)
        y -= self.y_base_size
        self._build_color_pool('g', self.grey_marbles, y)
        y -= self.y_base_size
        self._build_color_pool('w', self.white_marbles, y)

    def _setup_camera(self):
        """Setup camera position and orientation based on board size."""
        # Get camera configuration for this board size
        config = self.CAMERA_CONFIG.get(self.rings, self.CAMERA_CONFIG[37])
        center_pos = config['center_pos']
        cam_dist = config['cam_dist']
        cam_height = config['cam_height']

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
        self.wb = WaterNode(self, -15, -10, 15, 10, 0,
                            # anim: vx, vy, scale, skip
                            anim=LVector4(0.0245, -0.0122, 1.5, 1),
                            distort=LVector4(0.2, 0.05, 0.8, 0.2))  # (0, 0, .5, 0))

    def _build_players_marble_pool(self):
        self.player_pools = {
            1: self._make_marble_dict(),
            2: self._make_marble_dict()
        }
        self.player_pool_coords = {
            1: [],
            2: []
        }

        # Find corner positions dynamically based on board size
        # Top-left corner: first letter, highest number in that column
        # Top-right corner: last letter, highest number in that column
        first_letter = self.letters[0]
        last_letter = self.letters[-1]

        # Find the highest row for each corner
        top_left_positions = [pos for pos in self.pos_to_coords.keys() if pos.startswith(first_letter)]
        top_right_positions = [pos for pos in self.pos_to_coords.keys() if pos.startswith(last_letter)]

        # Get positions with max number (top of board)
        # Rightmost non-empty value in the first row
        first_row = self.pos_array[0]
        top_right = first_row[first_row != ''][-1]

        # First value in the first row
        top_left = first_row[0]

        logger.debug(f"Board: top_left={top_left}, top_right={top_right}")
        # Get adjacent positions to calculate board direction vectors
        # Find a neighbor position to determine the board's edge direction
        second_letter = self.letters[1]
        second_from_top_left = [pos for pos in self.pos_to_coords.keys()
                                if pos.startswith(second_letter) and int(pos[1:]) == int(top_left[1:])]

        second_last_letter = self.letters[-2]
        second_from_top_right = [pos for pos in self.pos_to_coords.keys()
                                 if pos.startswith(second_last_letter) and int(pos[1:]) == int(top_right[1:])]

        # Rightmost non-empty value in the second row
        second_row = self.pos_array[1]
        second_from_top_right = second_row[second_row != ''][-2]

        # first value in second row
        second_from_top_left = second_row[1]

        logger.debug(f"Board: second_from_top_left={second_from_top_left}, second_from_top_right={second_from_top_right}")
        # Get coordinates
        tl_coord = self.pos_to_coords[top_left]
        tr_coord = self.pos_to_coords[top_right]

        # Use adjacent positions if they exist, otherwise use same row one position down
        if second_from_top_left:
            tl_adj = self.pos_to_coords[second_from_top_left]
        else:
            # Fallback: use position one row down in same column
            adj_pos = f"{first_letter}{int(top_left[1:])-1}"
            tl_adj = self.pos_to_coords.get(adj_pos, tl_coord)

        if second_from_top_right:
            tr_adj = self.pos_to_coords[second_from_top_right]
        else:
            adj_pos = f"{last_letter}{int(top_right[1:])-1}"
            tr_adj = self.pos_to_coords.get(adj_pos, tr_coord)

        # Calculate direction vectors from corner to adjacent position using numpy arrays
        tl_coord = np.array(tl_coord)
        tl_adj = np.array(tl_adj)
        tr_coord = np.array(tr_coord)
        tr_adj = np.array(tr_adj)
        player_pool_member_offset = np.array(self.player_pool_member_offset)

        d_ul = (tl_coord - tl_adj) * self.player_pool_offset_scale
        p_ul = tl_coord + d_ul

        # For player 2 (right side), we want the pool closer and further from camera
        # Use a smaller offset and add extra Y offset to move away from camera
        d_ur = (tr_coord - tr_adj) * self.player_pool_offset_scale
        p_ur = tr_coord + d_ur

        # Create 12 positions per player (6 + 6 with offset)
        for r in range(2):
            for i in range(6):
                pp1 = p_ul + (-d_ur * (i / 1.5)) + (-player_pool_member_offset * (r + 1))
                pp1[2] += 0.25
                self.player_pool_coords[1].append(tuple(pp1))

                pp2 = p_ur + (-d_ul * (i / 1.5)) + (player_pool_member_offset * (r + 1))
                pp2[2] += 0.25
                self.player_pool_coords[2].append(tuple(pp2))

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
        h_max = lambda xx: r_max - abs(self.letters.index(self.letters[xx]) - (r_max // 2))
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

                if pos != '':  # Only create pieces for non-empty positions
                    base_piece = BasePiece(self)
                    x = x_center + (k * self.x_base_size) - x_row_offset
                    y = y_center - y_row_offset
                    coords = (x, y, 0)
                    base_piece.set_pos(coords)
                    self.pos_to_base[pos] = base_piece
                    self.pos_to_coords[pos] = coords

                    # Create coordinate label if show_coords is enabled
                    if self.show_coords:
                        text_node = TextNode(f'label_{pos}')
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
        # point light
        p_light = DirectionalLight("p_light")
        p_node = self.render.attachNewNode(p_light)
        p_node.setPos(-12, -2, 12)  # Set the camera
        p_node.lookAt(0, 0, 0)
        p_light.setColor((1, 1, 1, 1))
        self.render.setLight(p_node)
        p_node.hide(BitMask32(1))

        # light 1
        s_light1 = DirectionalLight('s_light1')
        s_light1.setColor((.75, .75, .75, 1))

        s_node1 = self.render.attachNewNode(s_light1)
        s_node1.setPos(0, 0, 20)
        s_node1.lookAt(0, 0, 0)
        s_node1.hide(BitMask32(1))
        self.render.setLight(s_node1)

        # ambient light
        a_light = AmbientLight("a_light")
        a_light.setColor((.0125, .025, .035, 1.00))
        a_node = self.render.attachNewNode(a_light)
        a_node.hide(BitMask32(1))

        self.render.setLight(a_node)

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
                self.animation_queue.put({
                    'entity': base_piece,
                    'src': base_pos,
                    'dst': None,
                    'scale': None,
                    'duration': action_duration,
                    'defer': action_duration
                })
            self.removed_bases.append((base_piece, base_pos))

        # If there's a marble, remove it and add to player's captured pool
        if marble_color is not None and pos in self.pos_to_marble:
            captured_marble = self.pos_to_marble.pop(pos)
            src_coords = captured_marble.get_pos()

            # Add to player's captured pool
            capture_pool = self.player_pools[player.n][marble_color]
            capture_pool_length = sum([len(k) for k in self.player_pools[player.n].values()])
            capture_pool.append(captured_marble)

            if capture_pool_length >= len(self.player_pool_coords[player.n]):
                capture_pool_length = len(self.player_pool_coords[player.n]) - 1

            player_pool_coords = self.player_pool_coords[player.n][capture_pool_length]

            if action_duration == 0:
                captured_marble.set_pos(player_pool_coords)
                captured_marble.set_scale(self.captured_marble_scale)
            else:
                self.animation_queue.put({
                    'entity': captured_marble,
                    'src': src_coords,
                    'dst': player_pool_coords,
                    'scale': self.captured_marble_scale,
                    'duration': action_duration,
                    'defer': action_duration
                })

    def update_frozen_regions(self, frozen_position_strs):
        """Apply visual fade effect to rings in frozen isolated regions.

        Args:
            frozen_position_strs: Set or list of position strings (e.g., {'A1', 'B2'})
        """
        fade_alpha = 0.7  # Alpha value for frozen rings (0=invisible, 1=fully opaque)

        for pos_str in frozen_position_strs:
            # Apply fade to ring (base piece) only - marbles stay fully visible
            if pos_str in self.pos_to_base:
                base_piece = self.pos_to_base[pos_str]
                base_piece.model.setColorScale(1, 1, 1, fade_alpha)  # White tint with reduced alpha
                base_piece.model.setTransparency(TransparencyAttrib.MAlpha)  # Enable alpha transparency

    def show_marble_placement(self, player, action_dict, action_duration=0):
        """Place a marble on the board (PUT action only, no ring removal).

        Args:
            action_duration: Animation duration
        """
        action_marble_color = action_dict['marble']
        dst = action_dict['dst']
        dst_coords = self.pos_to_coords[dst]

        # add marble from pool
        pool = self.marble_pool[action_marble_color]
        if len(pool) == 0:
            pool = self.player_pools[player.n][action_marble_color]
        if len(pool) == 0:
            logger.error(f"No marbles available in pool for player {player.n}, color {action_marble_color}")
            return
        put_marble = pool.pop()
        mip = self.marbles_in_play[action_marble_color]
        src_coords = put_marble.get_pos()
        if put_marble not in [p for p, _ in mip]:
            mip.append((put_marble, src_coords))

        if action_duration == 0:
            put_marble.set_pos(dst_coords)
            put_marble.set_scale(self.board_marble_scale)
        else:
            self.animation_queue.put({
                'entity': put_marble,
                'src': src_coords,
                'dst': dst_coords,
                'scale': self.board_marble_scale,
                'duration': action_duration,
                'defer': 0
            })
        self.pos_to_marble[dst] = put_marble

    def show_ring_removal(self, action_dict, action_duration=0):
        """Remove a ring from the board (PUT action only).

        Args:
            action_dict: Action dictionary with 'remove' key
            action_duration: Animation duration (already scaled by controller)
        """
        base_piece_id = action_dict['remove']
        if base_piece_id != '':
            if base_piece_id in self.pos_to_base:
                base_piece = self.pos_to_base[base_piece_id]
                base_pos = base_piece.get_pos()
                if action_duration == 0:
                    base_piece.hide()
                else:
                    # Queue ring removal animation
                    removal_defer = action_duration
                    self.animation_queue.put({
                        'entity': base_piece,
                        'src': base_pos,
                        'dst': None,
                        'scale': None,
                        'duration': action_duration,
                        'defer': removal_defer
                    })

                self.removed_bases.append((base_piece, base_pos))

    def show_action(self, player, render_data, action_duration=0., action_result=None):
        """Visualize an action without highlights.

        Args:
            player: Player making the move
            render_data: RenderData value object containing action_dict
            action_duration: Animation duration
            action_result: ActionResult containing frozen positions (optional)
        """
        # Extract data from value objects
        action_dict = render_data.action_dict
        frozen_positions = action_result.newly_frozen_positions if action_result else None

        # Player 2: {'action': 'PUT', 'marble': 'g',              'dst': 'G2', 'remove': 'D0'}
        # Player 1: {'action': 'CAP', 'marble': 'g', 'src': 'G2', 'dst': 'E2', 'capture': 'b'}
        # Player 1: {'action': 'PASS'}
        action = action_dict['action']

        # PASS actions have no visual component
        if action == 'PASS':
            return

        action_marble_color = action_dict['marble']
        dst = action_dict['dst']
        dst_coords = self.pos_to_coords[dst]

        if action == 'PUT':
            # Call the split methods (duration already scaled by controller)
            self.show_marble_placement(player, action_dict, action_duration)
            self.show_ring_removal(action_dict, action_duration)

        elif action == 'CAP':
            src = action_dict['src']
            src_coords = self.pos_to_coords[src]
            cap = action_dict['cap']
            cap_coords = self.pos_to_coords[cap]
            captured_marble_color = action_dict['capture']
            action_marble = self.pos_to_marble.pop(src)
            captured_marble = self.pos_to_marble.pop(cap)
            self.pos_to_marble[dst] = action_marble
            if action_duration == 0:
                action_marble.set_pos(dst_coords)
            else:
                self.animation_queue.put({
                    'entity': action_marble,
                    'src': src_coords,
                    'dst': dst_coords,
                    'scale': self.board_marble_scale,
                    'duration': action_duration,
                    'defer': 0
                })

            # captured_marble.hide()
            capture_pool = self.player_pools[player.n][captured_marble_color]
            capture_pool_length = sum([len(k) for k in self.player_pools[player.n].values()])
            capture_pool.append(captured_marble)
            # print(capture_pool_length, self.player_pools[player.n])
            if capture_pool_length >= len(self.player_pool_coords[player.n]):
                logger.error(f"Capture pool length ({capture_pool_length}) exceeds available coords ({len(self.player_pool_coords[player.n])}) for player {player.n}")
                capture_pool_length = len(self.player_pool_coords[player.n]) - 1
            player_pool_coords = self.player_pool_coords[player.n][capture_pool_length]
            if action_duration == 0:
                captured_marble.set_pos(player_pool_coords)
                captured_marble.set_scale(self.captured_marble_scale)
            else:
                self.animation_queue.put({
                    'entity': captured_marble,
                    'src': cap_coords,
                    'dst': player_pool_coords,
                    'scale': self.captured_marble_scale,
                    'duration': action_duration,
                    'defer': action_duration
                })

    def reset_board(self):
        # 1. Clear animations FIRST (before resetting visuals)
        while not self.animation_queue.empty():
            self.animation_queue.get()
        self.current_animations = []

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
                marble.model.clearMaterial()  # Clear any highlight materials
                self.marble_pool[color].append(marble)
                marble.set_scale(self.pool_marble_scale)
                marble.set_pos(pos)

        # 5. Clear marbles_in_play dict (important - this accumulates otherwise!)
        self.marbles_in_play = self._make_marble_dict()

        # 6. Clear pos_to_marble dict (CRITICAL - stale entries prevent highlights!)
        self.pos_to_marble.clear()

        # 7. Rebuild player pools
        self._build_players_marble_pool()

        # 8. Recreate highlight state machine
        if self.show_moves:
            self.action_visualization_sequencer = ActionVisualizationSequencer(self)
        self._action_context = None

    def is_busy(self):
        """Check if renderer is busy with highlights or animations.

        Returns:
            True if state machine is active or animations are running
        """
        if self.action_visualization_sequencer and self.action_visualization_sequencer.is_active():
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
        self._set_action_context(player, render_data, action_result, task_delay_time, on_complete)

        if self.show_moves and self.action_visualization_sequencer:
            # Debug logging
            if render_data.action_dict['action'] == "CAP" and render_data.capture_moves:
                print(f"[Renderer] Passing {len(render_data.capture_moves)} capture moves to state machine:")
                for cm in render_data.capture_moves:
                    print(f"  - {cm['src']} -> {cm['dst']}")

            # Start highlighting - pass render_data and action_result to state machine
            self.action_visualization_sequencer.start(player, render_data, task_delay_time)
        else:
            # Direct visualization without highlights
            self.show_action(player, render_data, task_delay_time, action_result)

        # If no animations or highlights were queued, complete immediately
        self._complete_action_if_ready()

    def _apply_highlight(self, highlight_info):
        """Apply a highlight to the specified rings and/or marbles.

        Args:
            highlight_info: Dict with 'entities', 'duration', 'color', 'emission',
                          'start_time', 'end_time' (timing already set by update loop)
        """

        entities_list = highlight_info.get('entities', [])
        color = highlight_info.get('color', self.PLACEMENT_HIGHLIGHT_COLOR)
        emission = highlight_info.get('emission', self.PLACEMENT_HIGHLIGHT_EMISSION)

        # Store original materials and what type was highlighted
        original_materials = {}

        for pos_str in entities_list:
            # Try to highlight marble first, then ring
            entity = None
            entity_type = None

            if pos_str in self.pos_to_marble:
                # Highlight the marble at this position
                entity = self.pos_to_marble[pos_str]
                entity_type = 'marble'
            elif pos_str in self.pos_to_base:
                # Highlight the ring at this position
                entity = self.pos_to_base[pos_str]
                entity_type = 'ring'

            if entity is not None:
                original_mat = entity.model.getMaterial()

                # Store original material properties for blending
                if original_mat is not None:
                    original_color = original_mat.getBaseColor()
                    original_emission = original_mat.getEmission()
                    original_metallic = original_mat.getMetallic()
                    original_roughness = original_mat.getRoughness()
                else:
                    # Default properties if no material exists
                    original_color = LVector4(0.8, 0.8, 0.8, 1.0)
                    original_emission = LVector4(0.0, 0.0, 0.0, 1.0)
                    original_metallic = 0.9
                    original_roughness = 0.1

                original_materials[pos_str] = (
                    original_mat,
                    entity_type,
                    original_color,
                    original_emission,
                    original_metallic,
                    original_roughness
                )

        # Store original materials and target colors for later blending
        highlight_info['original_materials'] = original_materials
        highlight_info['target_color'] = color
        highlight_info['target_emission'] = emission

    def _clear_highlight(self, highlight_info):
        """Clear a highlight and restore original materials.

        Args:
            highlight_info: Dict with 'original_materials'
        """
        original_materials = highlight_info.get('original_materials', {})

        for pos_str, mat_info in original_materials.items():
            original_mat, entity_type, _, _, _, _ = mat_info

            # Get the entity based on type
            entity = None
            if entity_type == 'marble' and pos_str in self.pos_to_marble:
                entity = self.pos_to_marble[pos_str]
            elif entity_type == 'ring' and pos_str in self.pos_to_base:
                entity = self.pos_to_base[pos_str]

            if entity is not None:
                if original_mat is not None:
                    entity.model.setMaterial(original_mat, 1)
                else:
                    entity.model.clearMaterial()

    def queue_animation(self, anim_type='move', defer=0, **kwargs):
        """Add an animation to the unified queue.

        Args:
            anim_type: 'move' or 'highlight'
            defer: Delay before starting (seconds)
            **kwargs: Type-specific parameters
                For 'move': entity, src, dst, scale, duration
                For 'highlight': entities (or rings), duration, color, emission
        """
        anim_item = {
            'type': anim_type,
            'defer': defer,
            **kwargs
        }
        self.animation_queue.put(anim_item)

    def queue_highlight(self, rings, duration, color=None, emission=None, defer=0):
        """Add a highlight to the queue.

        Args:
            rings: List of position strings (e.g., ['A1', 'B2', 'C3'])
            duration: How long to show highlight in seconds
            color: LVector4 color (defaults to PLACEMENT_HIGHLIGHT_COLOR)
            emission: LVector4 emission (defaults to PLACEMENT_HIGHLIGHT_EMISSION)
            defer: Delay before starting (seconds)
        """
        self.queue_animation(
            anim_type='highlight',
            defer=defer,
            entities=rings,
            duration=duration,
            color=color if color is not None else self.PLACEMENT_HIGHLIGHT_COLOR,
            emission=emission if emission is not None else self.PLACEMENT_HIGHLIGHT_EMISSION
        )

    def is_animation_active(self):
        """Check if any animations (move or highlight) are active or queued."""
        return len(self.current_animations) > 0 or not self.animation_queue.empty()

    @property
    def current_action_result(self):
        """Return the in-flight action result (if any)."""
        if self._action_context:
            return self._action_context['action_result']
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
            logger.warning("Renderer action context replaced before completion; forcing completion.")
            self._complete_action()

        self._action_context = {
            'player': player,
            'render_data': render_data,
            'action_result': action_result,
            'task_delay_time': task_delay_time,
            'on_complete': on_complete,
        }

    def _complete_action_if_ready(self) -> None:
        """Invoke completion callback when highlights and animations finish."""
        if self._action_context is None:
            return
        if self.action_visualization_sequencer and self.action_visualization_sequencer.is_active():
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

        callback = context.get('on_complete')
        if callback:
            callback(context['player'], context['action_result'])

    def report_status(self, message: str) -> None:
        """Handle textual status reports for compatibility with composite renderers."""
        logger.info(message)
