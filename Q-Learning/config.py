import math
# user-defined parameters
min_explore_rate = 0.1
min_learning_rate = 0.1
max_episodes = 900
max_time_steps = 200
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0


x_limits = (-0.5, 0.5)
phi_limits = (-50, 50)

# how many buckets should each state variable have?
#(x, ẋ, θ, θ̇)
number_of_buckets = (1, 1, 6, 3)


visualisation_speed = 0.01

# saving address
file_name = 'weights/' + str(min_explore_rate) + '_' + str(number_of_buckets)