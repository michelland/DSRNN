import logging

import numpy as np

from crowd_sim.envs.utils.obstacle import Obstacle
from crowd_sim.envs.utils.obstacle import ObstacleRectangle
from crowd_sim.envs.utils.obstacle import ObstacleCircle


class Map(object):
    def __init__(self, config, path):
        # self.radius = config.getfloat('obstacles', 'radius')
        self.radius = 0.5
        self.grid = np.loadtxt(path, dtype=int)
        self.obstacles_circle = []
        self.obstacles_rectangle = []
        self.obstacle_num = 0
        self.size = int(path.split(sep='_')[2])
        for i in range(0,(2 * self.size) + 1):
            for j in range(0,(2 * self.size) + 1):
                if self.grid[i,j] == 1:
                    self.obstacles_circle.append(ObstacleCircle(j - self.size, self.size - i, self.radius))
                    # self.obstacle_num += 1
                elif self.grid[i,j] == 2:
                    self.obstacles_rectangle.append(ObstacleRectangle(j - self.size, self.size - i, self.radius))
                    self.obstacle_num += 1

    def print_info(self):
        info = "obstacles rectangles at positions :"
        for o in self.obstacles_rectangle:
            info += f" {o.get_position()}({type(o)}"
        print(info)

        info2 = "obstacles circulaires at positions :"
        for o in self.obstacles_circle:
            info2 += f" {o.get_position()}({type(o)}"
        print(info2)




