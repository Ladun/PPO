import pybullet
import os

from .scene_bases import Scene
from ..utils import get_model_data

class StadiumScene(Scene):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multiplayer = False
        self.stadium_loaded = False

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if not self.stadium_loaded:
            self.stadium_loaded = True
            
            self.ground_plane_mjcf = self._p.loadSDF(get_model_data("plane_stadium.sdf"))

            # print(f"ground plane: {self.ground_plane_mjcf}")/a
            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i,-1,lateralFriction=0.8, restitution=0.5)
                self._p.changeVisualShape(i,-1,rgbaColor=[1,1,1,0.8])
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,1)

