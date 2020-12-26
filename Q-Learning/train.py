import gym
import argparse
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def parse_input():

    parser = argparse.ArgumentParser(description='Inverses Pendel mit QLearning')
    parser.add_argument('-t,','--training',action='store_true',help='Switches to training mode. Default=false')
    parser.add_argument('-l','--learning_rate',type=float,help='Min Learning rate', default=0.1)
    parser.add_argument('-e', '--exploration_rate', type=float, help='Min Exploration rate', default=0.1)

    args = parser.parse_args()
    return args

def train():
    args = parse_input()

    env = gym.make('CartPole-v0')

    no_buckets = (1, 1, 6, 3)
    no_actions = env.action_space.n

    state_value_bounds = list(zip(env.observation_space.low,
                                  env.observation_space.high))
    state_value_bounds[1] = (-0.5, 0.5)
    state_value_bounds[3] = (-math.radians(50), math.radians(50))

    print(state_value_bounds)
    print(len(state_value_bounds))
    print(np.shape(state_value_bounds))
    print(state_value_bounds[0][0])

    # define q_value_table
    q_value_table = np.zeros(no_buckets + (no_actions,))

    # Q has 6 dimensions 1 x 1 x 6 x 3 x 2
    print(q_value_table)




if __name__ == "__train__":
    train()