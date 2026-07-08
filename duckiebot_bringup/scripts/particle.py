import math


class Particle:

    def __init__(self, x, y, theta):

        self.x = x
        self.y = y
        self.theta = theta

        self.weight = 1.0

    def as_pose(self):

        return self.x, self.y, self.theta