import logging
import math
import sys

from queue import SimpleQueue, Empty

import numpy as np
import simplepbr
from direct.showbase.ShowBase import ShowBase

from panda3d.core import AmbientLight, LVector4, BitMask32, DirectionalLight, WindowProperties, loadPrcFileData, Material

from renderer.water_node import WaterNode
from renderer.zertz_models import BasePiece, SkyBox, make_marble

logger = logging.getLogger(__name__)


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

    # Camera configuration per board size
    CAMERA_CONFIG = {
        37: {'center_pos': 'D4', 'cam_dist': 10, 'cam_height': 8},
        48: {'center_pos': 'D5', 'cam_dist': 10, 'cam_height': 10},
        61: {'center_pos': 'E5', 'cam_dist': 11, 'cam_height': 10}
    }

    def __init__(self, white_marbles=6, grey_marbles=8, black_marbles=10, rings=37):
        # Configure OpenGL version before initializing ShowBase
        loadPrcFileData("", "gl-version 3 2")
        super().__init__()

        props = WindowProperties()
        # props.setSize(1364, 768)
        # props.setSize(1024, 768)

        self.win.requestProperties(props)

        self.animation_queue = SimpleQueue()
        self.current_animations = {}
        self.pos_to_base = {}
        self.pos_to_marble = {}
        self.removed_bases = []
        self.pos_to_coords = {}
        self.pos_array = None

        # Highlight queue for move visualization
        self.highlight_queue = SimpleQueue()
        self.current_highlight = None  # {'rings': [...], 'end_time': t, 'original_materials': {...}}

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

        self._build_base()

        self.marble_pool = None
        self.marbles_in_play = None
        self.white_marbles = white_marbles
        self.black_marbles = black_marbles
        self.grey_marbles = grey_marbles
        self._build_marble_pool()

        self._build_players_marble_pool()

        # Setup camera after board is built so we can center on it
        self._setup_camera()

        self.task = self.taskMgr.add(self.update, 'zertzUpdate', sort=50)

    def update(self, task):
        # for m in self.marble_pool:
        #    m.model.setR(10*task.time)
        # entity, src_coords, dst_coords,
        #
        # self.board_marble_scale, action_duration

        # Process animation queue - add new animations
        while not self.animation_queue.empty():
            try:
                anim_info = self.animation_queue.get_nowait()
                entity = anim_info['entity']
                self.current_animations[entity] = {
                    'src': anim_info['src'],
                    'dst': anim_info['dst'],
                    'src_scale': entity.get_scale(),
                    'dst_scale': anim_info['scale'],
                    'duration': anim_info['duration'],
                    'insert_time': task.time,
                    'start_time': task.time + anim_info['defer']
                }
            except Empty:
                pass

        # Update all active animations
        to_remove = []
        for entity, anim_data in self.current_animations.items():
            src = anim_data['src']
            dst = anim_data['dst']
            src_scale = anim_data['src_scale']
            dst_scale = anim_data['dst_scale']
            duration = anim_data['duration']
            start_time = anim_data['start_time']
            elapsed_time = task.time - start_time
            if elapsed_time < 0:
                continue
            elif elapsed_time > duration:
                to_remove.append(entity)
                if dst_scale is not None:
                    entity.set_scale(dst_scale)
                if dst is not None:
                    entity.set_pos(dst)
                continue
            anim_factor = elapsed_time / duration

            if src_scale is not None and dst_scale is not None:
                sx = src_scale.x + (dst_scale - src_scale.x) * anim_factor
                entity.set_scale(sx)

            jump = True
            if dst is None:
                dx, dy, dz = src
                adx = -1 if dx < 0 else 1
                ady = -1 if dy < 0 else 1

                dx = max(10, abs(dx) * 10) * adx
                dy = max(4, abs(dy) * 4) * ady

                # dx = -7.5 if dx < 0 else 7.5

                fz = 1.5
                dst = (dx, dy, dz*fz)
                #anim_factor = anim_factor
                jump = False
                # entity.hide()
            #else:
            x, y, z = self.get_current_pos(anim_factor, dst, src, jump=jump)
            entity.set_pos((x, y, z))

        for entity in to_remove:
            self.current_animations.pop(entity)

        # Process highlight queue
        # Check if current highlight has expired
        if self.current_highlight is not None:
            if task.time >= self.current_highlight['end_time']:
                # Clear current highlight
                self._clear_highlight(self.current_highlight)
                self.current_highlight = None

        # Start next highlight if no current highlight and queue has items
        if self.current_highlight is None:
            try:
                highlight_info = self.highlight_queue.get_nowait()
                # Apply highlight immediately
                self._apply_highlight(highlight_info, task.time)
                self.current_highlight = highlight_info
            except Empty:
                pass

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

    def _build_base(self):
        self._init_pos_coords()

        # Get canonical board layout from game logic (single source of truth)
        from game.zertz_board import ZertzBoard
        self.pos_array = ZertzBoard.generate_standard_board_layout(self.rings)

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
        """Animate removal of an isolated ring (with or without marble)."""
        action_duration *= 0.45

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

    def show_marble_placement(self, player, action_dict, action_duration=0):
        """Place a marble on the board (PUT action only, no ring removal)."""
        action_marble_color = action_dict['marble']
        dst = action_dict['dst']
        dst_coords = self.pos_to_coords[dst]
        action_duration *= 0.45

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
        """Remove a ring from the board (PUT action only)."""
        action_duration *= 0.45
        base_piece_id = action_dict['remove']
        if base_piece_id != '':
            if base_piece_id in self.pos_to_base:
                base_piece = self.pos_to_base[base_piece_id]
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

    def show_action(self, player, action_dict, action_duration=0):
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
        action_duration *= 0.45

        if action == 'PUT':
            # Call the split methods
            self.show_marble_placement(player, action_dict, action_duration / 0.45)  # Undo the scaling since the methods apply it
            self.show_ring_removal(action_dict, action_duration / 0.45)

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
        for b, pos in self.removed_bases:
            b.set_pos(pos)
        self.removed_bases.clear()

        self._build_players_marble_pool()

        for color, marbles in self.marbles_in_play.items():
            for marble, pos in marbles:
                self.marble_pool[color].append(marble)
                marble.set_scale(self.pool_marble_scale)
                marble.set_pos(pos)

        while not self.animation_queue.empty():
            self.animation_queue.get()

        self.current_animations = {}

    def highlight_placement_positions(self, placement_array, board):
        """Highlight rings where marbles can be placed.

        Args:
            placement_array: numpy array (3, width², width²+1) of valid placements
            board: ZertzBoard instance to convert indices to position strings
        """
        # Extract unique destination positions
        placement_positions = np.argwhere(placement_array)
        unique_dests = set()
        for marble_idx, dst, rem in placement_positions:
            unique_dests.add(dst)

        # Highlight each valid placement position
        for dst_idx in unique_dests:
            dst_y = dst_idx // board.width
            dst_x = dst_idx % board.width
            pos_str = board.index_to_str((dst_y, dst_x))

            if pos_str and pos_str in self.pos_to_base:
                base_piece = self.pos_to_base[pos_str]
                original_mat = base_piece.model.getMaterial()
                self.highlighted_rings[pos_str] = original_mat

                # Create highlight material matching original properties
                highlight_mat = Material()
                if original_mat is not None:
                    highlight_mat.setMetallic(original_mat.getMetallic())
                    highlight_mat.setRoughness(original_mat.getRoughness())
                else:
                    highlight_mat.setMetallic(0.9)
                    highlight_mat.setRoughness(0.1)

                highlight_mat.setBaseColor(self.PLACEMENT_HIGHLIGHT_COLOR)
                highlight_mat.setEmission(self.PLACEMENT_HIGHLIGHT_EMISSION)
                base_piece.model.setMaterial(highlight_mat, 1)

    def highlight_removable_rings(self, removable_indices, board):
        """Highlight rings that can be removed.

        Args:
            removable_indices: List of board indices for removable rings
            board: ZertzBoard instance to convert indices to position strings
        """
        for rem_idx in removable_indices:
            rem_y = rem_idx // board.width
            rem_x = rem_idx % board.width
            pos_str = board.index_to_str((rem_y, rem_x))

            if pos_str and pos_str in self.pos_to_base:
                base_piece = self.pos_to_base[pos_str]
                original_mat = base_piece.model.getMaterial()
                self.highlighted_rings[pos_str] = original_mat

                # Create highlight material using same colors as placement
                highlight_mat = Material()
                if original_mat is not None:
                    highlight_mat.setMetallic(original_mat.getMetallic())
                    highlight_mat.setRoughness(original_mat.getRoughness())
                else:
                    highlight_mat.setMetallic(0.9)
                    highlight_mat.setRoughness(0.1)

                highlight_mat.setBaseColor(self.PLACEMENT_HIGHLIGHT_COLOR)
                highlight_mat.setEmission(self.PLACEMENT_HIGHLIGHT_EMISSION)
                base_piece.model.setMaterial(highlight_mat, 1)

    def clear_move_highlights(self):
        """Clear all move highlights and restore original materials."""
        for pos_str, original_mat in self.highlighted_rings.items():
            if pos_str in self.pos_to_base:
                base_piece = self.pos_to_base[pos_str]
                if original_mat is not None:
                    base_piece.model.setMaterial(original_mat, 1)
                else:
                    base_piece.model.clearMaterial()
        self.highlighted_rings.clear()

    def _apply_highlight(self, highlight_info, current_time):
        """Apply a highlight to the specified rings and/or marbles.

        Args:
            highlight_info: Dict with 'rings', 'duration', 'color', 'emission'
            current_time: Current task time
        """
        rings = highlight_info['rings']
        color = highlight_info.get('color', self.PLACEMENT_HIGHLIGHT_COLOR)
        emission = highlight_info.get('emission', self.PLACEMENT_HIGHLIGHT_EMISSION)
        duration = highlight_info['duration']

        # Store original materials and what type was highlighted
        original_materials = {}

        for pos_str in rings:
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
                original_materials[pos_str] = (original_mat, entity_type)

                # Create and apply highlight material
                highlight_mat = Material()
                if original_mat is not None:
                    highlight_mat.setMetallic(original_mat.getMetallic())
                    highlight_mat.setRoughness(original_mat.getRoughness())
                else:
                    highlight_mat.setMetallic(0.9)
                    highlight_mat.setRoughness(0.1)

                highlight_mat.setBaseColor(color)
                highlight_mat.setEmission(emission)
                entity.model.setMaterial(highlight_mat, 1)

        # Update highlight_info with calculated end time and original materials
        highlight_info['end_time'] = current_time + duration
        highlight_info['original_materials'] = original_materials

    def _clear_highlight(self, highlight_info):
        """Clear a highlight and restore original materials.

        Args:
            highlight_info: Dict with 'original_materials'
        """
        original_materials = highlight_info.get('original_materials', {})

        for pos_str, mat_info in original_materials.items():
            original_mat, entity_type = mat_info

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

    def queue_highlight(self, rings, duration, color=None, emission=None):
        """Add a highlight to the queue.

        Args:
            rings: List of position strings (e.g., ['A1', 'B2', 'C3'])
            duration: How long to show highlight in seconds
            color: LVector4 color (defaults to PLACEMENT_HIGHLIGHT_COLOR)
            emission: LVector4 emission (defaults to PLACEMENT_HIGHLIGHT_EMISSION)
        """
        highlight_info = {
            'rings': rings,
            'duration': duration,
            'color': color if color is not None else self.PLACEMENT_HIGHLIGHT_COLOR,
            'emission': emission if emission is not None else self.PLACEMENT_HIGHLIGHT_EMISSION
        }
        self.highlight_queue.put(highlight_info)

    def is_highlight_active(self):
        """Check if any highlights are active or queued."""
        return self.current_highlight is not None or not self.highlight_queue.empty()

    def clear_highlight_queue(self):
        """Clear all queued highlights and current highlight."""
        # Clear current highlight if exists
        if self.current_highlight is not None:
            self._clear_highlight(self.current_highlight)
            self.current_highlight = None

        # Empty the queue
        while not self.highlight_queue.empty():
            try:
                self.highlight_queue.get_nowait()
            except Empty:
                break
