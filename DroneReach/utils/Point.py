import random

import numpy as np
import math


class Point:

    def __init__(self, *position):
        self.x, self.y, self.z = position

        '''
        6 total discrete movement options (E W N S U D)
        '''
        self._action_to_velocity_dict = {
            0: np.array([1, 0, 0], int),
            1: np.array([-1, 0, 0], int),
            2: np.array([0, 1, 0], int),
            3: np.array([0, -1, 0], int),
            4: np.array([0, 0, 1], int),
            5: np.array([0, 0, -1], int)
        }

        self.all_drift_dir = [np.array([1, 0, 0], int),
                              np.array([-1, 0, 0], int),
                              np.array([0, 1, 0], int),
                              np.array([0, -1, 0], int),
                              np.array([0, 0, 1], int),
                              np.array([0, 0, -1], int)
                              ]

        self.drift_velocity = self.all_drift_dir[random.randint(0, 5)]
        # We will use random speed for drifting in discrete environment instead.
        # self.drift_velocity = np.random.normal(self.drift_speed, self.drift_scale, 3)
        self.non_drift_collided = False  # This is to indicate if the agent has collided with something's

        self.drift_path = [(self.x, self.y, self.z)]
        self.drift_velocity_recorder = [self.drift_velocity]
        self.velocity_recorder = []
        self.desired_velocity_recorder = []

    def __str__(self):
        return f"Point ({self.x}, {self.y}, {self.z})"

    def __sub__(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def get_location(self):
        return np.array([self.x, self.y, self.z])

    def _try_move(self, velocity, collision: bool, collision_recovery: bool, is_agent: bool):
        """
        Try moving in the given direction, without actually modifying object's coordinate
        This could be used to check if object is within bounds before moving.
        :param velocity: list or numpy array that can be unpacked to three integers, x, y, z
        :param is_agent: if the object concerned an agent/DroneReach or not.
        :param collision: if the object is colliding with others.
        :param: collision recovery: whether to perform collision recovery.
        :return: a numpy array of shape (1, 3) for new location after the move.
        :return: a numpy array of shape(1, 3) for new location should the move is executed.
        """
        if collision and is_agent and collision_recovery:
            # print("try: collision! reversing velocity!")
            velocity = - velocity
        new_x = self.x + velocity[0]
        new_y = self.y + velocity[1]
        new_z = self.z + velocity[2]
        return np.array([new_x, new_y, new_z])

    def _move(self, velocity, collision: bool, collision_recovery: bool, is_agent: bool):
        """
        Given a 3-D velocity vector, return the new location in numpy array.
        \n Implement naive collision recovery (reversing absolute velocity), and can be turned off and on.
        :param velocity: list or numpy array that can be unpacked to three integers, x, y, z
        :param is_agent: if the object concerned an agent/DroneReach or not.
        :param collision: if the object is colliding with others.
        :param: collision recovery: whether to perform collision recovery.
        :return: a numpy array of shape (1, 3) for new location after the move.
        """
        if collision and is_agent and collision_recovery:
            # print("collision! reversing velocity!")
            velocity = - velocity

        self.x += velocity[0]
        self.y += velocity[1]
        self.z += velocity[2]
        new_loc = np.array([self.x, self.y, self.z])

        # for agent
        self.velocity_recorder.append(velocity)
        # only for obstacles
        self.drift_path.append(tuple(new_loc))
        self.drift_velocity_recorder.append(self.drift_velocity)
        return new_loc

    def try_drift(self, choice=None):
        """
        Given a choice input, try to drift in the matching direction.
        If input is not provided, just try drift in original direction.
        A special API for obstacles.
        :param choice: an integer between 0 and 5.
        :return: the location after trying.
        """
        if choice is not None:
            drift_velocity = self._action_to_velocity_dict[choice]
        else:
            drift_velocity = self.drift_velocity
        return self._try_move(drift_velocity, collision=False, collision_recovery=False, is_agent=False)

    def drift(self, choice=None):
        """
        Simply move with a preset drift velocity
        :param: collision: whether the object is colliding with others.
        :param: collision_recovery: if recovering from collision is enabled.
        :param: is_agent: if the object concerned an agent or not.
        :return: a numpy array of shape(1, 3) for new location after the drift
        """
        if choice is not None:
            self.drift_velocity = self._action_to_velocity_dict[choice]
        return self._move(self.drift_velocity, collision=False, collision_recovery=False, is_agent=False)

    def obs_hit_borders(self, borders):
        """
        Re-spawn the position within, and sample new drft velocity
        :param borders: a numpy array contains the x, y, z borders for the environment.
        :return:a list contains all the valid next positions.
        """
        # curr_loc = [self.x, self.y, self.z]
        #
        # borders_check = ((self.x - borders[0]), (self.y - borders[1]), (self.z - borders[2]))

        self.x = min(borders[0] - 1, max(0, self.x + 1))
        self.y = min(borders[1] - 1, max(0, self.y + 1))
        self.z = min(borders[2] - 1, max(0, self.z + 1))

        self.drift_velocity = self.all_drift_dir[random.randint(0, 5)]

    def obs_hit_buildings(self):

        self.drift_velocity = self.all_drift_dir[random.randint(0, 5)]

    def try_action(self, choice=None, collision=False, collision_recovery=False):
        """
        Try to move given the choice, without actually updating object's coordinate
        :param collision_recovery: if the collision recovery is enabled.
        :param collision: if the object is colliding with others.
        :param choice: an integer between 0 and 5
        :return: new location **should** the move is executed.
        """
        if choice is not None:
            desired_velocity = self._action_to_velocity_dict[choice]
        else:
            desired_velocity = self._action_to_velocity_dict[random.randint(0, 5)]
        return self._try_move(desired_velocity, collision, collision_recovery, True)

    def action(self, choice=None, collision=False, collision_recovery=True):
        """
        Given a discrete choice (0 ~ 5), perform the corresponding move (up, down, right, etc)
        according to the built-in key-direction dictionary.
        If no choice is given, just perform a random move.
        This method should only be called by agent/DroneReach.
        :param collision: if the object is colliding with others.
        :param collision_recovery: if the collision recovery is enabled.
        :param choice: an integer between 0 and 5
        :return: new location after the move is executed.
        """
        if choice is not None:
            desired_velocity = self._action_to_velocity_dict[choice]
        else:
            desired_velocity = self._action_to_velocity_dict[random.randint(0, 5)]
        self.drift_velocity_recorder.append(desired_velocity)
        return self._move(desired_velocity, collision, collision_recovery, True)

    def vector(self, other):
        if self == other:
            return np.array([0, 0, 0])
        vect = other.get_location() - self.get_location()
        return vect / math.sqrt(sum(vect * vect))

    def copy(self):
        return Point(self.x, self.y, self.z)
