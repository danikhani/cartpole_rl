import gym
import argparse
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Select an action - explore vs exploit
# epsilon-greedy method
def select_action(state_value, explore_rate,env,q_value_table):
    if random.random() < explore_rate:
        action = env.action_space.sample()  # explore
    else:
        action = np.argmax(q_value_table[state_value])  # exploit
    return action


def select_explore_rate(x,min_explore_rate):
    # change the exploration rate over time.
    return max(min_explore_rate, min(1.0, 1.0 - math.log10((x + 1) / 25)))


def select_learning_rate(x,min_learning_rate):
    # Change learning rate over time
    return max(min_learning_rate, min(1.0, 1.0 - math.log10((x + 1) / 25)))


def bucketize_state_value(state_value,state_value_bounds,no_buckets):
    ''' Discretizes continuous values into fixed buckets'''
    # print('len(state_value):', len(state_value))
    bucket_indices = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:  # violates lower bound
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:  # violates upper bound
            bucket_index = no_buckets[i] - 1  # put in the last bucket
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i] - 1) * state_value_bounds[i][0] / bound_width
            scaling = (no_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * state_value[i] - offset))

        bucket_indices.append(bucket_index)
    return (tuple(bucket_indices))


def discretize(obs,state_value_bounds,no_buckets):
    ''' Does the same job as bucketize'''
    upper_bounds = [state_value_bounds[i][1] for i in range(len(obs))]
    lower_bounds = [state_value_bounds[i][0] for i in range(len(obs))]
    ratios = [(obs[i] + abs(lower_bounds[i])) /
              (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((no_buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(no_buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)
