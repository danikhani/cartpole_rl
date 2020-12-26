import gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import train_utils as t_utils
import config


def start_train():

    env = gym.make('CartPole-v0')
    env._max_episode_steps = config.max_time_steps

    # number of actions which are possible
    no_actions = env.action_space.n

    state_value_bounds = list(zip(env.observation_space.low,
                                  env.observation_space.high))
    state_value_bounds[1] = config.x_limits
    state_value_bounds[3] = (math.radians(config.phi_limits[0]), math.radians(config.phi_limits[1]))

    # define q_value_table
    q_value_table = np.zeros(config.number_of_buckets + (no_actions,))


    _DEBUG = False
    frames = []
    reward_per_episode = []
    time_per_episode = []
    avgtime_per_episode = []
    learning_rate_per_episode = []
    explore_rate_per_episode = []

    # train the system
    totaltime = 0
    for episode_no in range(config.max_episodes):

        explore_rate = t_utils.select_explore_rate(episode_no,config.min_explore_rate)
        learning_rate = t_utils.select_learning_rate(episode_no,config.min_learning_rate)

        learning_rate_per_episode.append(learning_rate)
        explore_rate_per_episode.append(explore_rate)

        # reset the environment while starting a new episode
        observation = env.reset()

        start_state_value = t_utils.bucketize_state_value(observation,state_value_bounds,config.number_of_buckets)
        previous_state_value = start_state_value

        done = False
        time_step = 0

        while not done:
            # env.render()
            action = t_utils.select_action(previous_state_value, explore_rate,env,q_value_table)
            observation, reward_gain, done, info = env.step(action)
            state_value = t_utils.bucketize_state_value(observation,state_value_bounds,config.number_of_buckets)
            best_q_value = np.max(q_value_table[state_value])

            # update q_value_table
            q_value_table[previous_state_value][action] += learning_rate * (
                    reward_gain + config.discount * best_q_value -
                    q_value_table[previous_state_value][action])

            previous_state_value = state_value

            if episode_no % 100 == 0 and _DEBUG == True:
                print('Episode number: {}'.format(episode_no))
                print('Time step: {}'.format(time_step))
                print('Previous State Value: {}'.format(previous_state_value))
                print('Selected Action: {}'.format(action))
                print('Current State: {}'.format(str(state_value)))
                print('Reward Obtained: {}'.format(reward_gain))
                print('Best Q Value: {}'.format(best_q_value))
                print('Learning rate: {}'.format(learning_rate))
                print('Explore rate: {}'.format(explore_rate))

            time_step += 1
            # while loop ends here

        if time_step >= config.solved_time:
            config.no_streaks += 1
        else:
            config.no_streaks = 0

        if config.no_streaks > config.streak_to_end:
            print('CartPole problem is solved after {} episodes.'.format(episode_no))
            break

        # data log
        if episode_no % 100 == 0:
            print('Episode {} finished after {} time steps'.format(episode_no, time_step))
        time_per_episode.append(time_step)
        totaltime += time_step
        avgtime_per_episode.append(totaltime / (episode_no + 1))
        # episode loop ends here

    env.close()



    # Plotting
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(time_per_episode)
    axes[0].plot(avgtime_per_episode)
    axes[0].set(ylabel='time per episode')
    axes[1].plot(learning_rate_per_episode)
    axes[1].plot(explore_rate_per_episode)
    axes[1].set_ylim([0, 1])
    axes[1].set(xlabel='Episodes', ylabel='Learning rate')
    plt.savefig(config.file_name + '.png')
    plt.show()

    # savomg training weight
    np.save(config.file_name, q_value_table)

