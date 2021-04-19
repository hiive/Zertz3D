import sys

import numpy as np
import simplepbr
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, PointLight, Spotlight, PerspectiveLens, LVector4, BitMask32, DirectionalLight

from renderer.water_node import WaterNode
from renderer.zertz_models import BasePiece, BlackBallModel, GrayBallModel, WhiteBallModel, SkyBox


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

    def __init__(self):
        super().__init__()

        self.pos_to_base = {}
        self.pos_to_coords = {}
        self.pos_array = None

        self.x_base_size = 0.8
        self.y_base_size = 0.7

        self.number_offset = (self.x_base_size / 2, self.y_base_size, 0)  # (0.4, 0.7)
        self.letter_offset = (self.x_base_size / 2, -self.y_base_size, 0)   # (0.4,  -0.7)

        self.letters = "ABCDEFGH"

        self.pool_offset_scale = 1.5
        self.pool1_member_offset = (-0.6, 0, 0)
        self.pool2_member_offset = (0.6, 0, 0)

        self.pipeline = simplepbr.init(enable_shadows=False)
        self.pipeline.use_330 = True

        self.accept('escape', sys.exit)  # Escape quits
        self.accept('aspectRatioChanged', self._setup_water)
        self.disableMouse()  # Disable mouse camera control
        self.camera.setPosHpr(0, -10, 8, 0, -40, 0)  # Set the camera

        # self.camera.setPosHpr(0, 0, 16, 0, 270, 0)  # Set the camera
        self.setupLights()  # Setup default lighting

        sb = SkyBox(self)

        self.wb = None
        self._setup_water()

        # anim: vx, vy, scale, skip

        self._build_base()

        self._build_players_marble_pool()

        self._ball_placement_test()

    def _setup_water(self):
        if self.wb is not None:
            self.wb.remove()
            self.wb.destroy()

        # distort: offset, strength, refraction factor (0 = perfect mirror, 1 = total refraction), refractivity
        # l_texcoord1.xy = vtx_texcoord0.xy * k_wateranim.z + k_wateranim.xy * k_time.x
        self.wb = WaterNode(self, -10, -4.5, 10, 8, 0,
                            # anim: vx, vy, scale, skip
                            anim=LVector4(0.0245, -0.0122, 1.5, 1),
                            distort=LVector4(0.2, 0.05, 0.8, 0.2)) #(0, 0, .5, 0))

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
        # 3 marbles of each color, or 4 white
        # marbles, or 5 grey marbles, or 6 black marbles wins the game.
        a4 = self.pos_to_coords['A4']
        b4 = self.pos_to_coords['B4']
        d7 = self.pos_to_coords['E7']
        d6 = self.pos_to_coords['E6']
        d_ul = _mul_tuple(_sub_tuple(a4, b4), self.pool_offset_scale)
        p_ul = _add_tuple(a4, d_ul)
        d_ur = _mul_tuple(_sub_tuple(d7, d6), self.pool_offset_scale)
        p_ur = _add_tuple(d7, d_ur)
        for i in range(5):
            p1x = f'P1_{i + 1}'
            p2x = f'P2_{i + 1}'

            bb1 = WhiteBallModel(self)
            pp1 = _add_tuple(p_ul, _mul_tuple(_neg_tuple(d_ur), i / 2.0))
            x, y, z = pp1
            pp1 = (x, y, z + 0.25)
            bb1.set_pos(pp1)

            bb2 = WhiteBallModel(self)
            pp2 = _add_tuple(p_ur, _mul_tuple(_neg_tuple(d_ul), i / 2.0))
            x, y, z = pp2
            pp2 = (x, y, z + 0.25)
            bb2.set_pos(pp2)
        for i in range(4):
            p1x = f'P1_{i + 1}'
            p2x = f'P2_{i + 1}'

            bb1 = BlackBallModel(self)
            pp1 = _add_tuple(_add_tuple(p_ul, _mul_tuple(_neg_tuple(d_ur), i / 2.0)), self.pool1_member_offset)
            x, y, z = pp1
            pp1 = (x, y, z + 0.25)
            bb1.set_pos(pp1)

            bb2 = BlackBallModel(self)
            pp2 = _add_tuple(_add_tuple(p_ur, _mul_tuple(_neg_tuple(d_ul), i / 2.0)), self.pool2_member_offset)
            x, y, z = pp2
            pp2 = (x, y, z + 0.25)
            bb2.set_pos(pp2)
        self.marble_pool = []
        # for i in range(3):
        #    self.marble_pool.append()

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

    def setupLights(self):
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
