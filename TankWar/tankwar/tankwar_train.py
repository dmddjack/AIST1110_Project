#!/usr/bin/env python3

# Code source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import gc
import random
from collections import deque
from itertools import islice
from time import gmtime, strftime, time

import gym
import gym_tankwar
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
from cmdargs import args


def timer(start_time, progress, total) -> int:
    """
    Use time.time() to get the start time. Print the current progress and the estimated time.
    Return value updates the progress variable. Written for ESTR 2018 project.
    """

    print(f"Progress: {progress}/{total} ({progress/total*100:.2f}%)")
    print(f"Time elapsed: {strftime('%H:%M:%S', gmtime(time() - start_time))}")
    print(f"Estimated time: {strftime(f'%H:%M:%S', gmtime((time() - start_time) / (progress / total)))}")

    # return progress + 1


class RLModel:
    def __init__(self, env: gym.Env, state_shape, action_shape, train_episodes: int, seed: int | None = None):
        # Initialize variables that determine the behaviour of the searching of the action space
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.01

        self.env = env
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.train_episodes = train_episodes
        self.seed = seed

    def run(self):
        self.rewards, self.epsilons = [], []

        # Initialize the two models
        self.main_model = self._agent()
        self.target_model = self._agent()

        # Copy main_model's weights to target_model
        self.target_model.set_weights(self.main_model.get_weights())

        # Print the summary of the target model
        print(self.target_model.summary())

        self.replay_memory = deque(maxlen=20_000)

        steps_to_update_target_model = 0
        time_intvl = 1 * args.fps
        total_score, total_steps = 0, 0
        start_time = time()
        for episode in range(1, 1 + self.train_episodes):
            total_training_rewards = 0
            state, info = self.env.reset()
            # state, info = self.env.reset(seed=args.seed)

            terminated, truncated = False, False
            reward_interval = deque(maxlen=time_intvl)
            reward_interval_shoot = deque(maxlen=time_intvl)
            step = 0
            while not (terminated or truncated):
                steps_to_update_target_model += 1
                step += 1
                random_num = np.random.rand()
                if random_num <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_reshaped = state.reshape((1, state.shape[0]))
                    predicted = self.main_model.predict(state_reshaped, verbose=0)
                    action = np.argmax(predicted)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                #
                # if action == 4 or action == 9:
                #     reward += -.2
                # print(action)
                reward_interval_shoot.append(0)
                reward_interval.append(reward)
                reward_mean = np.array(reward_interval).mean()
                self.replay_memory.append([state, action, reward, new_state, terminated])
                if info["bullet lifetime"] is not None:
                    # print(reward_interval)
                    if info["bullet lifetime"] <= time_intvl:
                        reward_interval_shoot[-info["bullet lifetime"]] += 500
                    else:
                        self.replay_memory[-info["bullet lifetime"]][2] += 500

                if step > time_intvl:
                    self.replay_memory[-time_intvl][2] = reward_mean + reward_interval_shoot[0]
                if steps_to_update_target_model % 15 == 0 or (terminated or truncated):
                    self._train(terminated)
                # try:
                #     print(self.replay_memory[-time_intvl][1:3]) if self.replay_memory[-time_intvl][1] > 4 else None
                # except IndexError:
                #     pass
                state = new_state
                total_training_rewards += reward

                if terminated or truncated:
                    for i in range(1, min(time_intvl, step) + 1):
                        self.replay_memory[-i][2] = np.array(list(islice(reward_interval, i-1, None))).mean() + \
                                                    reward_interval_shoot[i-1]
                    self.rewards.append(total_training_rewards)
                    self.epsilons.append(self.epsilon)

                    # Save the target model for every 25 episodes
                    if episode % 25 == 0:
                        self.save(episode)

                    timer(start_time, episode, self.train_episodes)
                    print(f"Total training rewards = {total_training_rewards:<8.1f} at episode {episode:<{len(str(args.train_episodes))}d} "
                          f"with score = {info['score']:<2d}, steps = {info['steps']}")
                    total_score += info['score']
                    total_steps += info['steps']
                    if steps_to_update_target_model >= 400:
                        self.target_model.set_weights(self.main_model.get_weights())
                        steps_to_update_target_model = 0

                    break

            # Garbage collection for memory issue
            gc.collect()
            keras.backend.clear_session()

            print("Epsilon:", self.epsilon)
            print("=" * 40)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * episode)

        self.env.close()
        print(f"Avg score: {total_score/self.train_episodes}, Avg steps: {total_steps/self.train_episodes}")

    def _agent(self):
        learning_rate = 0.001

        init = tf.keras.initializers.HeUniform(seed=self.seed)
        model = keras.Sequential()
        model.add(
            keras.layers.Dense(
                128,
                input_shape=self.state_shape, 
                activation='relu', 
                kernel_initializer=init
            )
        )
        model.add(
            keras.layers.Dense(
                64,
                activation='relu',
                kernel_initializer=init
            )
        )
        model.add(
            keras.layers.Dense(
                64,
                activation='relu',
                kernel_initializer=init
            )
        )
        model.add(
            keras.layers.Dense(
                32,
                activation='relu',
                kernel_initializer=init
            )
        )
        model.add(
            keras.layers.Dense(
                self.action_shape, 
                activation='linear', 
                kernel_initializer=init
            )
        )
        model.compile(
            loss=tf.keras.losses.Huber(), 
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
            metrics=['accuracy']
        )

        return model

    def _train(self, terminated):
        learning_rate = 0.7
        discount_factor = 0.618
        batch_size = 512
        min_replay_size = 1_000

        if len(self.replay_memory) < min_replay_size:
            return
        # start_inx = random.randint(0, len(self.replay_memory) - batch_size)
        # mini_batch = list(islice(self.replay_memory, start_inx, start_inx+batch_size))
        mini_batch = random.sample(self.replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.main_model.predict(current_states, verbose=0)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        x, y = [], []
        for idx, (state, action, reward, new_state, terminated) in enumerate(mini_batch):
            if not terminated:
                max_future_q = reward + discount_factor * np.max(future_qs_list[idx])
            else:
                max_future_q = reward
            
            current_qs = current_qs_list[idx]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            x.append(state)
            y.append(current_qs)

        self.main_model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0, shuffle=True)

    # def get_qs(self, model, state, step):
    #     return model.predict(state.reshape([1, state.shape[0]]))[0]

    def save(self, episode: int):
        self.target_model.save(f"models/model_diff_{args.difficulty}_epi_{episode}.h5")

    def plot(self):
        fig = plt.figure()

        ax1 = fig.add_subplot(211)
        ax1.plot(np.arange(1, self.train_episodes + 1), self.rewards)
        ax1.set_title("Rewards over all episodes in training")
        ax1.set_ylabel("Reward")

        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(1, self.train_episodes + 1), self.epsilons)
        ax2.set_title("Epsilons over all episodes in training")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Epsilon")

        plt.show()


def main():
    # Make a directory to store target models if necessary
    if not os.path.isdir("models"):
        os.mkdir("models")

    assert args.mode != "human_rand"

    env = gym.make(
        "gym_tankwar/TankWar-v0", 
        render_mode=args.mode, 
        starting_hp=args.starting_hp,
        difficulty=args.difficulty,
        full_enemy=args.full_enemy,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)

    env.action_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    my_model = RLModel(
        env, 
        env.observation_space.shape, 
        env.action_space.n, 
        args.train_episodes, 
        args.seed
    )
    my_model.run()
    my_model.save(args.train_episodes)
    my_model.plot()


if __name__ == "__main__":
    main()
