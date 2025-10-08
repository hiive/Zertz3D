import math
import sys

from queue import SimpleQueue, Empty

import numpy as np
import simplepbr
from direct.showbase.ShowBase import ShowBase

from panda3d.core import AmbientLight, PointLight, Spotlight, PerspectiveLens, LVector4, BitMask32, DirectionalLight, \
    WindowProperties

from renderer.water_node import WaterNode
from renderer.zertz_models import BasePiece, BlackBallModel, GrayBallModel, WhiteBallModel, SkyBox, make_marble


class ZertzRenderer(ShowBase):

    def __init__(self, white_marbles=6, grey_marbles=8, black_marbles=10, rings=37):
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

        self.x_base_size = 0.8
        self.y_base_size = 0.7
        self.pool_marble_scale = 0.25
        self.board_marble_scale = 0.35
        self.captured_marble_scale = 0.9 * self.board_marble_scale

        self.number_offset = (self.x_base_size / 2, self.y_base_size, 0)  # (0.4, 0.7)
        self.letter_offset = (self.x_base_size / 2, -self.y_base_size, 0)  # (0.4,  -0.7)

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

        self.player_pool_offset_scale = 0.8
        self.player_pool_member_offset = (0.6, 0, 0)

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

        # self._ball_placement_test()

    def update(self, task):
        # for m in self.marble_pool:
        #    m.model.setR(10*task.time)
        # entity, src_coords, dst_coords,
        #
        # self.board_marble_scale, action_duration

        to_put_back = []
        while not self.animation_queue.empty():
            try:
                anim_info = self.animation_queue.get_nowait()
                # print(anim_info)
                entity, src, dst, scale, duration, defer = anim_info
                #if entity in self.current_animations:
                    # if it's already being animated, put it back in queue
                    # to_put_back.append(anim_info)
                    #pass
                #else:
                self.current_animations[entity] = (src, dst, entity.get_scale(), scale, duration, task.time,
                                                   task.time + defer)
            except Empty:
                pass

        for anim_info in to_put_back:
            self.animation_queue.put(anim_info)

        to_remove = []
        for entity, anim_data in self.current_animations.items():
            src, dst, src_scale, dst_scale, duration, insert_time, start_time = anim_data
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
        from game.zertz_board import ZertzBoard

        # Set the center position of the board
        center_pos = "D4"
        if self.rings == ZertzBoard.MEDIUM_BOARD_48:
            center_pos = "D5"
        elif self.rings == ZertzBoard.LARGE_BOARD_61:
            center_pos = "E5"

        # Get the actual 3D coordinates of the center position
        if center_pos in self.pos_to_coords:
            center_x, center_y, center_z = self.pos_to_coords[center_pos]
        else:
            # Fallback to origin if we can't find the center
            center_x, center_y, center_z = 0, 0, 0

        # Adjust camera distance and height based on board size
        cam_dist = 10  # Distance from board (Y axis)
        cam_height = 8  # Height above board (Z axis)
        if self.rings == ZertzBoard.MEDIUM_BOARD_48:
            cam_dist = 10
            cam_height = 10
        elif self.rings == ZertzBoard.LARGE_BOARD_61:
            cam_dist = 11
            cam_height = 10

        # Position camera to look at the board center
        # Camera is positioned behind (negative Y) and above (positive Z) the center point
        cam_x = center_x
        cam_y = center_y - cam_dist
        cam_z = center_z + cam_height

        self.camera.setPos(cam_x, cam_y, cam_z)
        # Point camera at the board center
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

    def _ball_placement_test(self):
        gb = GrayBallModel(self)
        gb.set_pos(self.pos_to_coords['A1'])
        bb = BlackBallModel(self)
        bb.set_pos(self.pos_to_coords['E7'])
        bb = BlackBallModel(self)
        bb.set_pos(self.pos_to_coords['E6'])
        wb = WhiteBallModel(self)
        wb.set_pos(self.pos_to_coords['A4'])
        wb = WhiteBallModel(self)
        wb.set_pos(self.pos_to_coords['B4'])
        # """

        # """
        gray_ball = GrayBallModel(self)
        gray_ball.set_pos(self.pos_to_coords['F3'])
        white_ball = WhiteBallModel(self)
        white_ball.set_pos(self.pos_to_coords['D4'])

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

        print(f"DEBUG: top_left, top_right", top_left, top_right)
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

        print(f"DEBUG: second_from_top_left, second_from_top_right", second_from_top_left, second_from_top_right)
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

            print(self.pos_array[i])

    def setup_lights(self):
        # point light
        p_light = DirectionalLight("p_light")
        p_node = self.render.attachNewNode(p_light)
        p_node.setPos(-12, -2, 12)  # Set the camera
        p_node.lookAt(0, 0, 0)
        p_light.setColor((1, 1, 1, 1))
        self.render.setLight(p_node)
        p_node.hide(BitMask32(1))

        # spot light 1
        s_light1 = DirectionalLight('s_light1')
        s_light1.setColor((.75, .75, .75, 1))
        # lens1 = PerspectiveLens()
        # lens1.setNearFar(0, 30)
        # s_light1.setLens(lens1)
        # # s_light1.setShadowCaster(True, 1024, 1024)
        s_node1 = self.render.attachNewNode(s_light1)
        s_node1.setPos(0, 0, 20)
        s_node1.lookAt(0, 0, 0)
        s_node1.hide(BitMask32(1))
        self.render.setLight(s_node1)

        # spot light 2
        """
        s_light2 = Spotlight('s_light2')
        s_light2.setColor((1, 1, 1, 1))
        s_light2.setLens(PerspectiveLens())
        s_node2 = self.render.attachNewNode(s_light2)
        s_node2.setPos(-20, -20, 20)
        s_node2.lookAt(0, 0, 0)
        self.render.setLight(s_node2)
        """

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
                self.animation_queue.put((base_piece, base_pos, None, None, action_duration, action_duration))
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
                self.animation_queue.put((captured_marble, src_coords, player_pool_coords,
                                          self.captured_marble_scale, action_duration, action_duration))

    def show_action(self, player, action_dict, action_duration=0):
        # Player 2: {'action': 'PUT', 'marble': 'g',              'dst': 'G2', 'remove': 'D0'}
        # Player 1: {'action': 'CAP', 'marble': 'g', 'src': 'G2', 'dst': 'E2', 'capture': 'b'}
        print(f'Player {player.n}: {action_dict}')

        action = action_dict['action']
        action_marble_color = action_dict['marble']
        dst = action_dict['dst']
        dst_coords = self.pos_to_coords[dst]
        action_duration *= 0.45

        if action == 'PUT':
            # remove piece
            base_piece_id = action_dict['remove']
            if base_piece_id != '':
                base_piece = self.pos_to_base[base_piece_id]
                base_pos = base_piece.get_pos()
                # base_piece.hide()
                if action_duration == 0:
                    base_piece.hide()
                else:
                    self.animation_queue.put((base_piece, base_pos, None, None, action_duration, action_duration))
                self.removed_bases.append((base_piece, base_pos))
            # add marble from pool
            pool = self.marble_pool[action_marble_color]
            if len(pool) == 0:
                pool = self.player_pools[player.n][action_marble_color]
            if len(pool) == 0:
                print('THIS IS A BUG!!!')
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
                self.animation_queue.put((put_marble, src_coords, dst_coords,
                                          self.board_marble_scale, action_duration, 0))
            self.pos_to_marble[dst] = put_marble

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
                self.animation_queue.put((action_marble, src_coords, dst_coords,
                                          self.board_marble_scale, action_duration, 0))

            # captured_marble.hide()
            capture_pool = self.player_pools[player.n][captured_marble_color]
            capture_pool_length = sum([len(k) for k in self.player_pools[player.n].values()])
            capture_pool.append(captured_marble)
            # print(capture_pool_length, self.player_pools[player.n])
            if capture_pool_length >= len(self.player_pool_coords[player.n]):
                print('THIS IS A BUG')
                capture_pool_length = len(self.player_pool_coords[player.n]) - 1
            player_pool_coords = self.player_pool_coords[player.n][capture_pool_length]
            if action_duration == 0:
                captured_marble.set_pos(player_pool_coords)
                captured_marble.set_scale(self.captured_marble_scale)
            else:
                self.animation_queue.put((captured_marble, cap_coords, player_pool_coords,
                                          self.captured_marble_scale, action_duration, action_duration))

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

        """        
        for _, marble_sets in self.player_pools.items():
            for color, marbles in self.marble_sets.items():
                for marble, pos in marbles:
                    self.marble_pool[color].append(marble)
                    marble.set_pos(pos)
                    marble.set_scale(self.pool_marble_scale)
        """
        pass
