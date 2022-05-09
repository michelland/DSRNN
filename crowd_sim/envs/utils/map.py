import logging
import random

import numpy as np

from crowd_sim.envs.utils.obstacle import Obstacle
from crowd_sim.envs.utils.obstacle import ObstacleRectangle
from crowd_sim.envs.utils.obstacle import ObstacleCircle


class Map(object):
    def __init__(self, radius, path, map_random = False):

        self.map_random = map_random
        # self.radius = config.getfloat('obstacles', 'radius')
        self.radius = radius
        self.size = int(path.split(sep='_')[2])
        self.obstacles_circle = []
        self.obstacles_rectangle = []
        self.obstacle_num = 0
        self.grid = None

        # if self.map_random:
        #     self.generate_random_map(6, 3)
        # else:
        #     self.generate_map_from_path(path)

    def generate_random_map(self, num_obstacles, radius_zone):
        self.obstacles_rectangle = []
        possible_positions = [(i, j) for i in range(-radius_zone, radius_zone + 1) for j in range(-radius_zone, radius_zone + 1)]
        ind = [i for i in range(len(possible_positions))]
        # random.seed(1)
        positions = np.random.choice(a=ind, size=num_obstacles, replace=False)
        for pos in positions:
            self.obstacles_rectangle.append(ObstacleRectangle(possible_positions[pos][0], possible_positions[pos][1], self.radius))

    def generate_map_from_path(self, path):
        self.grid = np.loadtxt(path, dtype=int)

        self.obstacles_rectangle = []
        for i in range(0, (2 * self.size) + 1):
            for j in range(0, (2 * self.size) + 1):
                if self.grid[i, j] == 1:
                    self.obstacles_circle.append(ObstacleCircle(j - self.size, self.size - i, self.radius))
                    # self.obstacle_num += 1
                elif self.grid[i, j] == 2:
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




