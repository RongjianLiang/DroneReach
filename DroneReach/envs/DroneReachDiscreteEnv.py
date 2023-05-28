import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

from DroneReach.utils.Point import Point


class DroneReachDiscrete(gym.Env):
    """

    """
    metadata = {"render_modes": ['human', 'save']}

    def __init__(self,
                 render_mode=None,
                 env_img_dir=None,
                 size: int = 10,
                 num_buildings: int = 10,
                 num_obstacles: int = 10,
                 step_goal_every: int = 5,
                 reach_threshold: int = 5,
                 binary_env: bool = False,
                 if_hetero_reward: bool = True,
                 ground_prox_penalty: float = 0.2,
                 collision_penalty: float = 0.25,
                 borders_penalty: float = 0.3,
                 move_penalty: float = 0.01,
                 resolve_cost: float = 0.5,
                 goal_prox_reward: float = 0.8
                 ):
        # environment information

        self.collision_penalty_queue = [0.]
        self.step_goal_every = step_goal_every
        self.reach_threshold = reach_threshold
        self.size = size  # the size of our environment
        self.max_flight_height = size - 2
        self.num_buildings = num_buildings
        self.num_obstacles = num_obstacles
        self.if_collision_recovery = True

        # rendering
        # print("init env...episode number cleared")
        self.render_mode = render_mode
        self.env_img_dir = env_img_dir
        self.env_eps_img_dir = ""
        self.video_frame = 0
        self.episode_number = 0

        # rewards configuration
        self.binary_env = binary_env
        self.if_hetero_reward = if_hetero_reward
        self.ground_prox_penalty = ground_prox_penalty
        self.collision_penalty = collision_penalty
        self.borders_penalty = borders_penalty
        self.move_penalty = move_penalty
        self.resolve_cost = resolve_cost
        self.goal_prox_reward = goal_prox_reward

        # create some numpy arrays expressing observation spaces
        point_loc_arr = np.full(3, size - 1, dtype=int)
        obstacles_loc_arr = np.full((size, size, size), 1, dtype=int)
        buildings_loc_arr = np.full((size, size, size), 1, dtype=int)

        # creating envs

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiDiscrete(point_loc_arr),
                "goal": spaces.MultiDiscrete(point_loc_arr),
                "obstacles_map": spaces.MultiDiscrete(obstacles_loc_arr),
                "buildings_map": spaces.MultiDiscrete(buildings_loc_arr)
            }
        )
        self._generate_envs()

        self.action_space = spaces.Discrete(6)

        # other info for debugging
        self.path = [self.agent[0].get_location()]
        self.episode_step = 0
        self.reach_reward_queue = [0.]
        self.move_penalty_queue = [0.]
        self.borders_penalty_queue = [0.]
        self.resolved_cost_queue = [0.]

    def _generate_envs(self):
        """
        A function to regenerate the terrains, obstacles, agent, and goal.
        \n 1. Generate terrain map and pass it to attribute.
        \n 2. Generate agent by emtpy blocks iterator.
        \n 3. Generate drones map the contains the agent's location.
        \n 4. Generate goals map that contains goals position.
        \n 5. Generate ...
        :return:
        """
        size = self.size
        self.terrain_map = self.generate_terrain(size)
        empty_blocks_iter = self._empty_blocks(self.terrain_map)
        self.agent = [Point(*next(empty_blocks_iter)), Point(*next(empty_blocks_iter))]
        self.drones_map = np.zeros((size, size, size), dtype=int)
        self.goals_map = np.zeros((size, size, size), dtype=int)
        self.obstacles_map = np.zeros((size, size, size), dtype=int)

        self.drones_map[tuple(self.agent[0].get_location())] = 1
        self.goals_map[tuple(self.agent[1].get_location())] = 1
        self.obstacles_list = [Point(*next(empty_blocks_iter)) for i in range(self.num_obstacles)]
        # for obs in self.obstacles_list:
        #     self.obs_curr_loc.append(obs.get_location())

        # create new dict to store transitions
        # add this dict to all dict
        # self.obstacle_eps_pos = {}
        # self.obstacle_all_pos[f"episode_{self.episode_number}"] = self.obstacle_eps_pos
        # obs_id = 0
        # for obs in self.obstacles_list:
        #     self.obstacles_map[tuple(obs.get_location())] = 1
        #     self.obstacle_eps_pos[f"obstacle_{obs_id}"] = list(obs.get_location())
        #     obs_id += 1

    def generate_terrain(self, size):
        terrain = np.zeros((size, size, size), dtype=int)

        if self.ground_prox_penalty and not self.binary_env:
            for i in range(size // 2):
                terrain[:, :, i] = self.ground_prox_penalty * (size // 2 - i) / (size // 2)

        for i in range(self.num_buildings):
            while True:
                # Generate random numbers in intervals of 0.5
                x, y = np.random.randint(0, size * 2, 2) / 2
                # Generate random building height
                z = np.random.randint(0, self.max_flight_height)
                # Check if existing buildings exist. If so, regenerate. Otherwise, keep building.
                if np.all(terrain[math.floor(x):math.ceil(x) + 1, math.floor(y):math.ceil(y) + 1, 0:z] != 1):
                    if not self.binary_env:
                        terrain[math.floor(x) - 1:math.ceil(x) + 2, math.floor(y) - 1:math.ceil(y) + 2, 0:z + 1] \
                            = terrain[math.floor(x) - 1:math.ceil(x) + 2, math.floor(y) - 1:math.ceil(y) + 2, 0:z + 1] \
                            .clip(min=0.5)
                    terrain[math.floor(x):math.ceil(x) + 1, math.floor(y):math.ceil(y) + 1, 0:z].fill(1)
                    break

        return terrain

    def _get_obs(self):
        goal_in = self.agent[0].get_location() - self.agent[0].get_location()
        observation = {"agent": self.agent[0].get_location(),
                       "buildings_map": self.terrain_map,
                       "goal": goal_in,
                       "obstacles_map": self.obstacles_map
                       }
        return observation

    def _get_info(self):
        info = {"current step:": self.episode_step,
                "collision penalty": self.collision_penalty_queue[-1],
                "reach reward:": self.reach_reward_queue[-1],
                "move penalty:": self.move_penalty_queue[-1],
                "borders penalty:": self.borders_penalty_queue[-1],
                "resolve cost:": self.resolved_cost_queue[-1]
                }
        # getting info on agent's collision
        # if self.collision_flag:
        #     info["agent collision step"] = self.episode_step
        # getting info on obstacle's position
        # obs_id = 0
        # obs_curr_loc = {}
        # for obs in self.obstacles_list:
        #     obs_curr_loc[f"obstacle_{obs_id}"] = obs.get_location()
        #     obs_id += 1
        # info["obstacle location"] = obs_curr_loc

        return info

    def _empty_blocks(self, occupied):
        """
        Return an iterator of empty blocks (represented by np.array([x, y, z])).
        :param occupied: a 3-D numpy array-like bitmap indicating which position are occupied
        :return: an iterator on empty blocks
        """
        empty_blocks = [np.array([x, y, z])
                        for x in range(self.size)
                        for y in range(self.size)
                        for z in range(self.size) if
                        occupied[x, y, z] != 1]
        random.shuffle(empty_blocks)
        return iter(empty_blocks)

    # TODO: fix reset function
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the terrain map
        self._generate_envs()

        # Reset the other variable
        # self.steps_of_agent_collision = []
        self.episode_step = 0
        self.video_frame = 0
        self.episode_number += 1
        if self.render_mode == 'save':
            eps_dir = f"episode-{self.episode_number}"
            print(f"\nenv reset for recording..")
            self.env_eps_img_dir = os.path.join(self.env_img_dir, eps_dir)
            print(f"creating dir: {self.env_eps_img_dir}\n")
            # try:
            os.mkdir(self.env_eps_img_dir)

        self.path = [self.agent[0].get_location()]
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _generate_bool_map(self):
        terrain_bool_map = (self.terrain_map == 1)
        obstacles_bool_map = (self.obstacles_map == 1)
        drones_bool_map = (self.drones_map == 1)
        return terrain_bool_map, obstacles_bool_map, drones_bool_map

    def agent_within_bounds(self, location):
        x, y, z = location
        return (0 <= x < self.size) and (0 <= y < self.size) and (0 <= z < self.size)

    def obs_within_bounds(self, location):
        x, y, z = location
        return 0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size

    def resolve_agent_bounce(self, t_bool_map, o_bool_map, curr_loc):
        agent = self.agent[0]
        valid_action = []
        for action in range(6):
            if self.agent_within_bounds(agent.try_action(action)):
                valid_action.append(action)

        resolve_action = []
        for resolve in valid_action:
            next_loc = agent.try_action(resolve)
            if not self._check_agent_collision(t_bool_map, o_bool_map, next_loc):
                resolve_action.append(resolve)

        resolution = random.choice(resolve_action)
        resolved_loc = agent.action(resolution, False, False)
        self.drones_map[tuple(curr_loc)] = 0
        self.drones_map[tuple(resolved_loc)] = 1
        return resolution

    def step_drone(self, action, curr_loc, collision, t_bool_map, o_bool_map):
        """
        if drone's next trying loc is outside of bounds, simply raise the flag, return curr loc, and let the drone holds.
        :param action: an integer between 0 and 5.
        :param curr_loc: numpy array
        :param collision: result of collision check between obstacles.
        :param t_bool_map: as is.
        :param o_bool_map: as is
        :return: next location after stepping, and a flag for if resolve routine is used.
        """
        try_next_loc = self.agent[0].try_action(action, collision=collision,
                                                collision_recovery=self.if_collision_recovery)
        if not self.agent_within_bounds(try_next_loc):
            # the bouncing get agent out of bound...
            # calling more advanced method to revolve the bounced direction...
            # we can not step obstacles now...
            # and this move is hold...how about assign a resolve cost?
            next_loc = self.resolve_agent_bounce(t_bool_map, o_bool_map, curr_loc)
            return next_loc, True
        # print(f"step, trying next loc: {try_next_loc}")
        if self._check_agent_collision(t_bool_map, o_bool_map, try_next_loc):
            next_loc = curr_loc
        else:
            self.agent[0].action(action, collision=collision, collision_recovery=self.if_collision_recovery)
            next_loc = try_next_loc
        # print(f"the next loc is {next_loc}")
        self.drones_map[tuple(curr_loc)] = 0
        self.drones_map[tuple(next_loc)] = 1
        self.path.append(next_loc)
        return next_loc, False

    def _step_goal(self, terrain_bool_map):
        """
        drift the goal to a valid and correct but random position.
        :param terrain_bool_map: as is.
        :return: goal next location.
        """
        goal = self.agent[1]
        curr_loc = goal.get_location()
        # print(f"goal curr loc: {curr_loc}")
        valid_drift = []
        for choice in range(6):
            if self.obs_within_bounds(goal.try_drift(choice)):
                valid_drift.append(choice)
        # and we do not want the goal to drift inside the buildings!
        # as for obstacles...the influence should be small.
        correct_drift = []
        for pos_choice in valid_drift:
            next_loc = goal.try_drift(pos_choice)
            if not terrain_bool_map.astype(int)[tuple(next_loc)]:
                correct_drift.append(pos_choice)

        # sample the random drift
        action = random.choice(correct_drift)
        next_goal_loc = goal.action(action, False, False)
        # print(f"goal drifting to {next_goal_loc}")
        self.goals_map[tuple(curr_loc)] = 0
        self.goals_map[tuple(next_goal_loc)] = 1
        return next_goal_loc

    def _step_obstacles(self, terrain_bool_map):
        """implement the stepping obstacles logic,
        and obstacles are step-able just like agent does.
        Collision is not checked between obstacles.
        When hit the bounds, a naive re-spawning is done.
        :return: none.
        """
        obs_id = 0
        for obs in self.obstacles_list:
            curr_loc = obs.get_location()
            self.obstacles_map[tuple(curr_loc)] = 0

            next_original_loc = obs.try_drift()

            # checking all possible next loc
            valid_choice = []
            for choice in range(6):
                if self.obs_within_bounds(obs.try_drift(choice)):
                    # print(f"-- obs: id {obs_id} valid next loc{obs.try_drift(choice)}")
                    valid_choice.append(choice)
                # then check for buildings
            correct_choice = []
            for pos_choice in valid_choice:
                next_loc = obs.try_drift(pos_choice)
                if not terrain_bool_map.astype(int)[tuple(next_loc)]:
                    correct_choice.append(pos_choice)
                    # print(f"** obs: id {obs_id} correct next loc{obs.try_drift(pos_choice)}")

            # determine if the original choice is correct
            change = True
            # print(f"++ obs: id {obs_id} all correct move: {correct_choice}")
            for i in correct_choice:
                # print(f"this is fine: {obs.try_drift(i)}")
                if all(np.equal(next_original_loc, obs.try_drift(i))):
                    # print(f"we have: {obs.try_drift(i)} equal to {next_original_loc}")
                    change = False

            if change:
                action = random.choice(correct_choice)
                next_loc = obs.drift(action)
            else:
                next_loc = obs.drift()

            obs_id += 1
            self.obstacles_map[tuple(next_loc)] = 1

    def _check_agent_collision(self, terrain_bool_map, obstacles_bool_map, next_loc):
        """
        Given current bool maps, determine given position could lead to a collision with buildings, obstacles.
        :param terrain_bool_map:
        :param obstacles_bool_map:
        :return: result of collision check
        """
        result = False
        # TODO: need to check borders before accessing the map
        collision_bitmap = (terrain_bool_map | obstacles_bool_map).astype(int)
        if collision_bitmap[tuple(next_loc)] == 1:
            result = True
            self.collision_flag = True
        return result

    def step(self, action):
        """
        """
        # -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]
        self.episode_step += 1
        terminated = False
        truncated = False
        reach = False
        collision = False
        run_into_border = False
        resolved = False

        # step the obstacles first
        terrain_bool_map = (self.terrain_map == 1)
        self._step_obstacles(terrain_bool_map)
        terrain_bool_map, obstacles_bool_map, drones_bool_map = self._generate_bool_map()

        if self.episode_step % self.step_goal_every == 0:
            goal_loc = self._step_goal(terrain_bool_map)
        else:
            goal_loc = self.agent[1].get_location()

        curr_loc = self.agent[0].get_location()

        next_original_pos = self.agent[0].try_action(action)

        # First check if within
        within = self.agent_within_bounds(next_original_pos)
        if within:
            # print(f"given curr_loc: {curr_loc}, and next org loc: {next_original_pos}, action is {action}")
            collision = self._check_agent_collision(terrain_bool_map, obstacles_bool_map, next_original_pos)
            next_loc, resolved = self.step_drone(action, curr_loc, collision, t_bool_map=terrain_bool_map,
                                                 o_bool_map=obstacles_bool_map)
        else:
            # print(f"not within bounds! you are at {curr_loc}, trying to move to {next_original_pos}")
            """Agent would remain current pos if next pos not within"""
            next_loc = curr_loc
            run_into_border = True

        discrete_steps = np.absolute(next_loc - goal_loc)
        # print(f"curr_loc: {curr_loc}, goal_loc: {goal_loc}")
        # print(f"--d-steps: {discrete_steps}, threshold: {self.reach_threshold} , "
        #       f"if reached: {discrete_steps.sum() <= self.reach_threshold}")
        reach_reward = self.calc_reach_stray_reward(discrete_steps)
        reward = reach_reward - self.move_penalty - \
                 collision * self.collision_penalty - \
                 run_into_border * self.borders_penalty - resolved * self.resolve_cost

        self.reach_reward_queue.append(reach_reward)
        self.move_penalty_queue.append(self.move_penalty)
        self.borders_penalty_queue.append(run_into_border * self.borders_penalty)
        self.resolved_cost_queue.append(resolved*self.resolve_cost)
        self.collision_penalty_queue.append(collision * self.collision_penalty)

        observation = self._get_obs()
        info = self._get_info()

        self.video_frame += 1

        if self.render_mode == 'save':
            self.render()

        return observation, reward, terminated, truncated, info

    def calc_reach_stray_reward(self, steps_to_goal):
        """
        :param steps_to_goal: A 3D numpy array, for number of steps at x, y, z to reach goal location.
        :return: calculated rewards
        """
        steps_threshold = self.reach_threshold
        total_steps = steps_to_goal.sum()
        if total_steps <= steps_threshold:
            reward = self.goal_prox_reward  # we need to cover the cost of moving towards target somehow...
        else:
            reward = 0
        return reward

    def render(self):

        elev = 60
        azim = 45
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True)

        terrain = self.terrain_map == 1
        drones = self.drones_map == 1
        goals = self.goals_map == 1
        obstacles = self.obstacles_map == 1

        voxel_arr = terrain | drones | goals | obstacles
        colors = np.empty(terrain.shape, dtype=object)
        colors[terrain] = '#7A88CCC0'
        colors[drones] = '#FFD65DC0'
        colors[goals] = '#607D3BC0'
        colors[obstacles] = '#FDA4BAC0'
        ax.voxels(voxel_arr, facecolors=colors, shade=True)
        if self.render_mode == 'save':
            # par_dir = self.env_eps_img_dir
            # image_name = f"-{self.episode_number}"
            # image_path = os.path.join(par_dir, image_name)
            par_dir = self.env_eps_img_dir
            image_name = f"{self.video_frame}.png"
            image_path = os.path.join(par_dir, image_name)
            plt.savefig(image_path, format='png')
            plt.close()
            return
        elif self.render_mode == 'human':
            plt.show()
        else:
            pass

    def close(self):
        pass
