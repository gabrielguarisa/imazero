import random
import numpy as np


class Particle:
    def __init__(
        self,
        initial_position,
        initial_velocity,
        lower_bound,
        upper_bound,
        local_acceleration,
        global_acceleration,
    ):
        self.position = initial_position.copy()
        self.velocity = initial_velocity.copy()
        self.best_position = initial_position.copy()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.local_acceleration = local_acceleration
        self.global_acceleration = global_acceleration
        self.best_score = 0

    def update_velocity(self, inertia_weight, global_best_position, d):
        r_p = random.uniform(0, 1)
        r_g = random.uniform(0, 1)
        self.velocity[d] = (
            inertia_weight * self.velocity[d]
            + self.local_acceleration * r_p * (self.best_position[d] - self.position[d])
            + self.global_acceleration
            * r_g
            * (global_best_position[d] - self.position[d])
        )

        return self.velocity

    def update_position(self, d):
        self.position[d] = self.position[d] + self.velocity[d]

        if self.position[d] < self.lower_bound:
            self.position[d] = self.lower_bound
        elif self.position[d] > self.upper_bound:
            self.position[d] = self.upper_bound
        return self.position[d]

    def disperse(self, d, rows, cols):
        """[summary]
            5 | 6 | 7
            0 | X | 1
            4 | 3 | 2
        Arguments:
            d {[type]} -- [description]
            rows {[type]} -- [description]
            cols {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        index_movement = [
            (-1, 0),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, -1),
            (0, -1),
            (1, -1),
        ]
        r_ = self.get_position(d) / cols
        c_ = self.get_position(d) % cols

        while True:
            if cols == 1:
                rand_c, rand_r = index_movement[random.randint(0, 1)]
            else:
                rand_c, rand_r = index_movement[random.randint(0, 7)]

            if (
                r_ + rand_r > 0
                and r_ + rand_r < rows
                and c_ + rand_c > 0
                and c_ + rand_c < cols
            ):
                r_ = r_ + rand_r
                c_ = c_ + rand_c
                break

        self.position[d] = r_ + c_ * cols

        return self

    def get_position(self, d=None):
        if d == None:
            return self.position

        return self.position[d]

    def get_best_position(self):
        return self.best_position

    def get_velocity(self):
        return self.velocity

    def update_best_score(self, new_score):
        if new_score > self.best_score:
            self.best_score = new_score
            self.best_position = self.position.copy()

    def get_best_score(self):
        return self.best_score
