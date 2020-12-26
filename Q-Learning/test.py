import gym
import argparse
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

import train_utils as t_utils



def test():
    q_value_table = np.load('file_name.npy')
    print(q_value_table)

    #make the gym
    env = gym.make('CartPole-v0')
    env._max_episode_steps = 400

    #env = gym.envs.register(id='CartPole-v0',max_episode_steps=250,      # MountainCar-v0 uses 200)
    #env = gym.make('CartPole-v0')
    # how many buckets should each state variable have?
    no_buckets = (1, 1, 6, 3)
    # number of actions which are possible
    no_actions = env.action_space.n

    #boundings for each state variable:
    state_value_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_value_bounds[1] = (-0.5, 0.5)
    state_value_bounds[3] = (-math.radians(50), math.radians(50))

    min_explore_rate = 0.1
    min_learning_rate = 0.1
    max_episodes = 1000
    max_time_steps = 200
    streak_to_end = 120
    solved_time = 500
    discount = 0.99
    no_streaks = 0


    # reset the environment while starting a new episode
    observation = env.reset()

    done = False
    time_step = 0
    while not done:
        print(time_step)
        env.render()
        time.sleep(.01)
        state_value = t_utils.bucketize_state_value(observation, state_value_bounds, no_buckets)
        action = np.argmax(q_value_table[state_value])
        observation, reward_gain, done, info = env.step(action)

        time_step += 1
        # while loop ends here

    if time_step >= solved_time:
        no_streaks += 1
    else:
        no_streaks = 0

    env.close()
'''    
    for episode_no in range(max_episodes):

        # reset the environment while starting a new episode
        observation = env.reset()

        done = False
        time_step = 0
        print(episode_no)
        while not done:
            print(time_step)
            env.render()
            time.sleep(.01)
            state_value = t_utils.bucketize_state_value(observation, state_value_bounds, no_buckets)
            action = np.argmax(q_value_table[state_value])
            observation, reward_gain, done, info = env.step(action)

            time_step += 1
            # while loop ends here

        if time_step >= solved_time:
            no_streaks += 1
        else:
            no_streaks = 0

        if no_streaks > streak_to_end:
            print('CartPole problem is solved after {} episodes.'.format(episode_no))
            break

'''





def play_episodes(env,get_policy_values,num_episodes=10, render=False):
    """
    Play some episodes using trained policy .
    Args:
        num_episodes: number of episodes to play
        render: wheter to renver video
    """
    for i_episode in range(num_episodes):
        # instansiating the environment
        observation = env.reset()
        for t in range(1000):
            # uncomment this is you want to see the rendering
            # env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                avg_time = avg_time + t
                if t > max_time:
                    max_time = t
                    print(max_time)
                # print("Episode finished after {} timesteps".format(t+1))
                break
        # resetting the enviroment
        env.reset()
        '''
    for i in range(num_episodes):
        rewards = []
        s = env.reset()
        for _ in range(1000):
            if render:
                env.render()
            action_probs = F.softmax(get_policy_values(s), dim=-1)
            sampler = Categorical(action_probs)
            a = sampler.sample()
            log_prob = sampler.log_prob(a)
            new_s, r, done, _ = env.step(a.item())

            rewards.append(r)
            s = new_s
            if done:
                print("Episode {} finished with reward {}".format(i + 1, np.sum(rewards)))
                break

'''
if __name__ == "__main__":
    test()