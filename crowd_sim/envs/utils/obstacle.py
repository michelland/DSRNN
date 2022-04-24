import logging
import abc
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.state import ObstacleState


class Obstacle(object):
    def __init__(self, px, py, radius):
        """
        Base class for obstacles
        """
        self.px = px
        self.py = py
        self.radius = radius
        self.shape = ''

    def print_info(self):
        logging.info(f"Obstacle is in position ({self.px},{self.py})")

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    @abc.abstractmethod
    def get_observable_state(self):
        """
        Returns obstacle's observable state
        """
        return


class ObstacleCircle(Obstacle):
    def __init__(self, px, py, radius):
        super().__init__(px, py, radius)
        self.shape = 'circle'

    def get_observable_state(self):
        return ObservableState(self.px, self.py, 0, 0, self.radius)


class ObstacleRectangle(Obstacle):
    def __init__(self, px, py, radius):
        super().__init__(px, py, radius)
        self.shape = 'rectangle'
        self.vertices = []
        self.vertices.append((px - 0.5, py + 0.5))
        self.vertices.append((px + 0.5, py + 0.5))
        self.vertices.append((px + 0.5, py - 0.5))
        self.vertices.append((px - 0.5, py - 0.5))
        # self.vertices.append((px - 0.5, py + 0.5))
        # self.vertices.append((px - 0.5, py - 0.5))
        # self.vertices.append((px + 0.5, py - 0.5))
        # self.vertices.append((px + 0.5, py + 0.5))

    def get_observable_state(self):
        return ObstacleState(self.shape, self.px, self.py, self.radius, vertices=self.vertices)
