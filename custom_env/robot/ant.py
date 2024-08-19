
from .robot_bases import MJCFBasedRobot
from .walker_base import WalkerBase
from ..utils import get_model_data

class Ant(WalkerBase, MJCFBasedRobot):
    def __init__(self):
        MJCFBasedRobot.__init__(self, get_model_data("ant.xml"), "torso", action_dim=8,  obs_dim=28)
        WalkerBase.__init__(self, power=2.5,
                                  foot_list=['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot'])

    def alive_bonus(self, z, pitch):
        return +1 if z >0.26 else -1