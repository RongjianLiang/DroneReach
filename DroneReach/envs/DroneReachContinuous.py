import copy
import heapq
import random

import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

# TODO: figure out what these constants are for...
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_SP = "speed"
_DIR = "direction"
_ACC = "acceleration"


class DroneReachContinuous(CUDAEnvironmentContext):

    def __init__(
            self,
            num_drones: int = 1,
            num_goals: int = 1,
            num_obstacles: int = 10,
            env_size: int = 10,
            env_height: int = 20,
            env_max_building_height: int = 18,
            env_difficulty: int = 1,
            starting_location_x=None,
            starting_location_y=None,
            starting_location_z=None,
            seed=None,
            base_speed: float = 1.0,
            max_drone_speed: float = 1.0,
            max_goal_speed: float = 1.0,
            max_obstacle_speed: float = 1.0,
            max_acceleration: float = 1.0,
            min_acceleration: float = -1.0,
            num_acceleration_levels: int = 10,
            edge_hit_penalty: float = -0.0,
            use_full_observation: bool = False,
            drone_obs_range: bool = 0.02,
            tagging_distance: float = 0.01,
            tag_reward_for_drone: float = 1.0,
            step_penalty_for_drone: float = -1.0,
            hit_penalty_for_drone: float = -1.0,
            tag_penalty_for_goal: float = -1.0,
            step_reward_for_goal: float = 0.0,
            env_backend="cpu",
    ):
        """

            :param num_drones: (int, optional): number of agents in the environment.
            Defaults to 1.
            :param num_goals:(int, optional): number of goals in the environment.
            Defaults to 1.
            :param num_obstacles:(int, optional): number of moving obstacles in the environment.
            Defaults to 10.
            :param env_size:(int, optional): size of the environment.
            Defaults to 10.
            :param env_difficulty:(int, optional): difficulty level of the environment, range between
            1 and 9 (both exclusive).
            Defaults to 1.
            :param env_height:(int, optional): height of the environment that agents and goals can act in.
            Defaults to 20.
            :param env_max_building_height:(int, optional): maximum height of buildings in the environment.
            Defaults to 18.
            :param starting_location_x:([ndarray], optional): [starting x locations of all agents].
            Defaults to None.
            :param starting_location_y:([ndarray], optional): [starting y locations of all agents].
            Defaults to None.
            :param starting_location_z:([ndarray], optional): [starting z locations of all agents].
            Defaults to None.
            :param seed:([type], optional): [seeding parameter].
            Defaults to None.
            :param base_speed:(float, optional): base speed for agents and goals.
            Defaults to 1.
            :param max_drone_speed:(float, optional): a base speed multiplier for drones.
            Defaults to 1.
            :param max_goal_speed:(float, optional): a base speed multiplier for goals.
            Defaults to 1.
            :param max_obstacle_speed:(float, optional): a base speed multiplier for obstacles.
            Defaults to 1.
            :param max_acceleration:(float, optional): the max acceleration.
            Defaults to 1.0.
            :param min_acceleration:(float, optional): the min acceleration.
            Defaults to -1.0.
            :param num_acceleration_levels:(int, optional): number of acceleration actions
            uniformly spaced between min and max acceleration.
            Defaults to 10.
            :param edge_hit_penalty:(float, optional): penalty for hitting the edge.
            Defaults t0 -0.0
            :param use_full_observation:(bool, optional): boolean indicating whether to include
            all the agents' data in the observation or just the ones within observation range.
            Defaults to False.
            :param drone_obs_range:(float, optional): range of drones' observation.
            This multiplies on top of the grid length.
            Defaults to 0.02.
            :param tagging_distance:(float, optional): margin between agent and goal to consider
            the goal has been 'tagged'. This multiplies on top of the grid length.
            Defaults to 0.01.
            :param tag_reward_for_drone:(float, optional): positive reward for the drone upon
            tagging the goal.
            Defaults to 1.0.
            :param step_penalty_for_drone:(float, optional): negative reward for every step the agents take.
            Defaults to -1.0.
            :param hit_penalty_for_drone:(float, optional): negative reward for drone's collision with obstacles.
            Defaults to -1.0.
            :param tag_penalty_for_goal:(float, optional): negative reward for goal getting tagged.
            Defaults to 0.0.
            :param step_reward_for_goal:(float, optional): positive reward for every step the goal is not
            getting tagged.
            Defaults to 0.0.
            :param env_backend:(string, optional): indicate whether to use the CPU or
            the GPU (either pycuda or numba) for stepping through the environment.
            Defaults to "cpu".
            """
        super().__init__()

        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)

        self.num_drones = num_drones
        self.num_goals = num_goals
        self.num_obstacles = num_obstacles

        self.num_agents = self.num_obstacles + self.num_goals + self.num_obstacles

        self.env_size = self.int_dtype(env_size)
        self.env_diagonal = self.env_size * np.sqrt(2)
        self.difficulty_level = env_difficulty
        self.env_height = env_height
        self.buildings_max_height = env_max_building_height

        self.edge_hit_penalty = self.float_dtype(edge_hit_penalty)

        # seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        # starting drone's ids list, and obstacle's ids list
        drones = self.np_random.choice(
            np.arange(self.num_agents), self.num_drones, replace=False
        )

        obstacles = self.np_random.choice(
            np.arange(self.num_obstacles), self.num_obstacles, replace=False
        )

        self.agent_type = {}
        self.drones = {}
        self.goals = {}
        self.obstacles = {}

        # the building dictionary would be populated when they are generated.
        # keys would be their locations (x, y), and values being height.
        self.buildings = {}

        # build the dictionary for each type of agents.
        # key would be the universal agent's id,
        for agents_id in range(self.num_agents):
            if agents_id in set(drones):
                self.agent_type[agents_id] = 1  # drones
                self.drones[agents_id] = True
            elif agents_id in set(obstacles):
                self.agent_type[agents_id] = 2  # obstacles
                self.obstacles[agents_id] = True
            else:
                self.agent_type[agents_id] = 3  # goals
                self.goals[agents_id] = True

        # spawn agents location.

        # first we generate the buildings
        self.city_base_map, _ = self.generate_city()
        # TODO: then populate the buildings dictionary
        for x in range(self.env_size):
            for y in range(self.env_size):
                if self.city_base_map[x, y]:
                    self.buildings[(x, y)] = self.city_base_map[x, y]
        #

    def generate_city(self):
        sec_num = 2
        city_base_map = np.zeros((self.env_size, self.env_size), dtype=self.int_dtype)

        mean_height_list = np.linspace(0.3, 0.9, 10).astype(self.float_dtype)
        std_height_list = np.geomspace(2, 0.4, 10).astype(self.float_dtype)

        idx = random.randint(0, len(mean_height_list) - 1)
        mean_height = mean_height_list[idx]
        std_height = std_height_list[idx]

        fill_per_sec = self.difficulty_level / 10

        h_mean = self.int_dtype(self.buildings_max_height * mean_height)
        h_std = self.float_dtype(h_mean * std_height)
        cell_size = self.int_dtype(self.env_size / sec_num)
        num = self.int_dtype(fill_per_sec * cell_size ** 2)

        a_min = np.zeros(num, dtype=self.int_dtype)
        a_max = np.zeros(num, dtype=self.int_dtype)

        for i in range(sec_num):
            for j in range(sec_num):
                building_height = np.clip(np.random.normal(h_mean, h_std, num),
                                          a_min, a_max)
                height_list = np.concatenate((building_height, np.zeros(cell_size ** 2 - num)))
                np.random.shuffle(height_list)
                block_height = np.reshape(height_list, (cell_size, cell_size))
                city_base_map[i * cell_size: (i + 1) * cell_size, j * cell_size: (j + 1) * cell_size] = block_height

        # calculate some statistics
        sum_height = city_base_map.sum()
        fill_ratio = sum_height / (self.env_size ** 2 * self.env_height)
        specific_fill_ratio = sum_height / (num * sec_num ** 2 * self.buildings_max_height)

        return city_base_map, (h_mean, h_std, fill_per_sec, fill_ratio, specific_fill_ratio)

    def base_map2bool_map(self):
        city_maps = np.zeros((self.env_size, self.env_size, self.env_height), dtype=self.int_dtype)
        city_base_map = self.city_base_map

        for i in range(self.env_height):
            city_maps[:, :, i] = (city_base_map >= 1)
            city_base_map -= 1
        return city_maps

    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
