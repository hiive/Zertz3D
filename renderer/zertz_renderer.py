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


def _mul_tuple(a, f):
    a1, a2, a3 = a
    return a1 * f, a2 * f, a3 * f


def _add_tuple(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return a1 + b1, a2 + b2, a3 + b3


def _sub_tuple(a, b):
    a1, a2, a3 = a
    b1, b2, b3 = b
    return a1 - b1, a2 - b2, a3 - b3


def _neg_tuple(a):
    a1, a2, a3 = a
    return -a1, -a2, -a3


class ZertzRenderer(ShowBase):

    def __init__(self, white_marbles=6, grey_marbles=8, black_marbles=10):
        #todo this needs to take number of rings.
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

        self.number_offset = (self.x_base_size / 2, self.y_base_size, 0)  # (0.4, 0.7)
        self.letter_offset = (self.x_base_size / 2, -self.y_base_size, 0)  # (0.4,  -0.7)

        #todo this needs to be adjusted for number of rings.
        self.letters = "ABCDEFGH"

        self.player_pool_offset_scale = 1.5
        self.player1_pool_member_offset = (-0.6, 0, 0)
        self.player2_pool_member_offset = (0.6, 0, 0)

        self.player_pools = None
        self.player_pool_coords = None

        self.pipeline = simplepbr.init()
        self.pipeline.enable_shadows = True
        self.pipeline.use_330 = True

        self.accept('escape', sys.exit)  # Escape quits
        self.accept('aspectRatioChanged', self._setup_water)
        self.disableMouse()  # Disable mouse camera control
        self.camera.setPosHpr(0, -10, 8, 0, -40, 0)  # Set the camera

        # self.camera.setPosHpr(0, 0, 16, 0, 270, 0)  # Set the camera
        self.setup_lights()  # Setup default lighting

        sb = SkyBox(self)

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

    def _setup_water(self):
        if self.wb is not None:
            self.wb.remove()
            self.wb.destroy()

        # distort: offset, strength, refraction factor (0 = perfect mirror, 1 = total refraction), refractivity
        # l_texcoord1.xy = vtx_texcoord0.xy * k_wateranim.z + k_wateranim.xy * k_time.x
        self.wb = WaterNode(self, -10, -4.5, 10, 8, 0,
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
        # 3 marbles of each color, or 4 white
        # marbles, or 5 grey marbles, or 6 black marbles wins the game.
        # TODO this is kind of a hacky way of positioning the marble pool, I think.
        a4 = self.pos_to_coords['A4']
        b4 = self.pos_to_coords['B4']
        d7 = self.pos_to_coords['E7']
        d6 = self.pos_to_coords['E6']
        d_ul = _mul_tuple(_sub_tuple(a4, b4), self.player_pool_offset_scale)
        p_ul = _add_tuple(a4, d_ul)
        d_ur = _mul_tuple(_sub_tuple(d7, d6), self.player_pool_offset_scale)
        p_ur = _add_tuple(d7, d_ur)
        for i in range(6):
            pp1 = _add_tuple(p_ul, _mul_tuple(_neg_tuple(d_ur), i / 2.0))
            x, y, z = pp1
            pp1 = (x, y, z + 0.25)
            self.player_pool_coords[1].append(pp1)

            pp2 = _add_tuple(p_ur, _mul_tuple(_neg_tuple(d_ul), i / 2.0))
            x, y, z = pp2
            pp2 = (x, y, z + 0.25)
            self.player_pool_coords[2].append(pp2)

        for i in range(6):
            pp1 = _add_tuple(_add_tuple(p_ul, _mul_tuple(_neg_tuple(d_ur), i / 2.0)), self.player1_pool_member_offset)
            x, y, z = pp1
            pp1 = (x, y, z + 0.25)
            self.player_pool_coords[1].append(pp1)

            pp2 = _add_tuple(_add_tuple(p_ur, _mul_tuple(_neg_tuple(d_ul), i / 2.0)), self.player2_pool_member_offset)
            x, y, z = pp2
            pp2 = (x, y, z + 0.25)
            self.player_pool_coords[2].append(pp2)

    def _init_pos_coords(self):
        self.pos_to_base.clear()
        self.pos_to_coords.clear()
        self.pos_array = None

    def _build_base(self):
        self._init_pos_coords()
        pos_array = []

        # 0  A4 B5 C6 D7
        # 1  A3 B4 C5 D6 E6
        # 2  A2 B3 C4 D5 E5 F5
        # 3  A1 B2 C3 D4 E4 F4 G4
        # 4     B1 C2 D3 E3 F3 G3
        # 5        C1 D2 E2 F2 G2
        # 6           D1 E1 F1 G1

        r_max = len(self.letters)
        is_even = r_max % 2 == 0
        h_max = lambda xx: r_max - abs(self.letters.index(self.letters[xx]) - (r_max // 2))
        r_min = h_max(0)
        if is_even:
            r_min += 1
        x_center = -(self.x_base_size / 2) * r_max / 2
        y_center = (self.y_base_size / 2) * r_max / 2
        for i in range(r_max):
            hh = h_max(i)
            ll = self.letters[:hh] if i < hh / 2 else self.letters[-hh:]
            nn_max = r_max - i
            nn_min = max(r_min - i, 1)
            x_row_offset = self.x_base_size / 2 * (h_max(i) - r_min)
            y_row_offset = self.y_base_size * i
            # print(y_row_offset)
            # print()
            pos_array.append([''] * r_max)
            for k in range(len(ll)):
                ix = min(k + nn_min, nn_max)
                lt = ll[k]
                pa = self.letters.find(lt)
                pos = f'{lt}{ix}'
                # if is_even and lt == self.letters[-1]:
                #    continue
                pos_array[i][pa] = pos
                base_piece = BasePiece(self)
                x = x_center + (k * self.x_base_size) - x_row_offset
                y = y_center - y_row_offset
                coords = (x, y, 0)
                base_piece.set_pos(coords)
                self.pos_to_base[pos] = base_piece
                self.pos_to_coords[pos] = coords
            print(pos_array[i])

        self.pos_array = np.array(pos_array)

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
            else:
                self.animation_queue.put((captured_marble, cap_coords, player_pool_coords,
                                          self.board_marble_scale, action_duration, action_duration))

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
