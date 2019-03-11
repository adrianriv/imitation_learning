# Hopper
import os
import pickle
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras import layers
import gym

import load_policy
from my_utils import plot_history
import tf_util


params = {'expert': 'experts/Hopper-v2.pkl', 'environment': 'Hopper-v2', 'env_max_timesteps': 500, 'render': False,
          'trajectories': 40, 'run_policy_trajectories': 10, 'training_epochs': 2, 'adam_lr': .001,
          'mlp_depth': 2, 'mlp_width': 64, 'mlp_activation': 'relu',
          'mlp_input_dim': 11, 'mlp_out_dim': 3, 'batch_size': 512}

f = open('expert_data/Hopper-v2.pkl', 'rb')

data = pickle.loads(f.read())  # keys: observations, actions


# Create a MLP to represent policy,
with tf.variable_scope('my_policy'):
    model = tf.keras.Sequential()
    # input layer
    model.add(layers.BatchNormalization(axis=1, center=True, scale=True,
                                        input_shape=(11,)))
    # hidden layers
    for _ in range(params['mlp_depth']):
        model.add(layers.Dense(64, activation=params['mlp_activation']))

    model.add(layers.Dense(params['mlp_out_dim'], activation='tanh'))

    model.add(layers.Reshape([1, params['mlp_out_dim']]))

model.compile(optimizer=tf.train.AdamOptimizer(params['adam_lr']), loss='mse',
              metrics=['mean_absolute_error', 'mean_squared_error'])


def train_my_policy(train_data, plot=False):
    history = model.fit(train_data['observations'], train_data['actions'], shuffle=True,
                        epochs=params['training_epochs'],
                        batch_size=params['batch_size'], steps_per_epoch=None, validation_split=0.2,
                        verbose=False)
    if plot:
        p = plot_history(history)
        p.show()


env = gym.make(params['environment'])
print('env.spec.timestep_limit', env.spec.timestep_limit)
print('env.action_space.high',env.action_space.high)
print('env.action_space.low',env.action_space.low)
max_steps = params['env_max_timesteps'] or env.spec.timestep_limit


def run_policy(policy, run_policy_trajectories=params['run_policy_trajectories'], render=params['render']):
    """returns observations"""
    observations = []
    actions = []

    for _ in range(run_policy_trajectories):
        totalr = 0
        steps = 0
        obs = env.reset()
        done = False

        while not done:
            action = policy(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0:
                #print("%i/%i" % (steps, max_steps))
                pass
            if steps >= max_steps:
                break

    return observations, totalr


# We now import expert policy
expert_policy = load_policy.load_policy(params['expert'])


def expand_dataset_with_expert_help(new_obs, data):
    expert_actions = expert_policy(new_obs)
    data['observations'] = np.vstack((data['observations'], new_obs))  # this will get slow as dataset grows
    data['actions'] = np.vstack((data['actions'], np.reshape(expert_actions, (-1, 1, 3))))


#data = {'observations': np.empty(data['observations'].shape), 'actions': np.empty(data['actions'].shape)}
with tf.Session():
    tf.global_variables_initializer()

    vec_rewards = []

    for i in range(params['trajectories']):
        print(data['observations'].shape, data['actions'].shape)
        #   train my_policy with current data
        if i >0:
            train_my_policy(data, plot=False)

        # let  my_policy play in the environment and get new_observations, collect current stats of policy

        new_obs, reward = run_policy(model.predict,render=False)
        #new_obs, reward = run_policy(expert_policy,render=False)
        print('reward', reward)

        new_obs = np.array(new_obs)

        # pass new_observations to expand data
        expand_dataset_with_expert_help(new_obs, data)

        _, reward = run_policy(model.predict, run_policy_trajectories=1, render=True)
        print('reward', reward)
        #run_policy(expert_policy, run_policy_trajectories=1, render=True)


