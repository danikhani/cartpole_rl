import gym
import argparse
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

import train_utils as t_utils
import config



def test():
    # load the npy file of q-table saved after training:
    q_value_table = np.load('file_name.npy')
    print(q_value_table)

    # make the gym
    env = gym.make('CartPole-v0')
    env._max_episode_steps = config.max_time_steps

    # how many buckets should each state variable have?
    no_buckets = config.number_of_buckets

    #boundings for each state variable:
    state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_value_bounds[1] = config.x_limits
    state_value_bounds[3] = (math.radians(config.phi_limits[0]), math.radians(config.phi_limits[1]))

    observation = env.reset()

    solved_time = config.max_time_steps
    for time_step in range(solved_time):
        #print(time_step)
        env.render()
        # speed of the renedering
        time.sleep(config.visualisation_speed)
        state_value = t_utils.bucketize_state_value(observation, state_value_bounds, no_buckets)
        action = np.argmax(q_value_table[state_value])
        # for logging
        observation, reward_gain, done, info = env.step(action)

    env.close()

if __name__ == "__main__":
    test()