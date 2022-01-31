"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control.mountain_car import MountainCarEnv


class MountainCarCustomEnv(MountainCarEnv):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.

    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).

    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right

        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.

    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.

    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.

    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, min_position=-1.2, max_position=0.6, max_speed=0.07, goal_position=0.5, goal_velocity=0, force=0.001, gravity=0.0025):
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position
        self.goal_velocity = goal_velocity

        self.force = force
        self.gravity = gravity

        self.low = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            self.low, self.high, dtype=np.float32
        )

        self.seed()
