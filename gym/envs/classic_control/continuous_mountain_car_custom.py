# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv


class Continuous_MountainCarCustomEnv(Continuous_MountainCarEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, min_action=-1.0, max_action=1.0, min_position=-1.2, max_position=0.6, max_speed=0.07, goal_position=0.45, goal_velocity=0, power=0.0015):
        self.min_action = min_action
        self.max_action = max_action
        self.min_position = min_position
        self.max_position = max_position
        self.max_speed = max_speed
        self.goal_position = goal_position # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = goal_velocity
        self.power = power

        self.low_state = np.array(
            [self.min_position, -self.max_speed], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()
        self.reset()

