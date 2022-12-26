#!/usr/bin/env python3

# Script reference: https://github.com/mswang12/minDQN/blob/main/minDQN.py

# Code source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import gc
import random
from collections import deque
from datetime import datetime
from itertools import islice
from time import gmtime, strftime, time

import gym
import gym_tankwar
import matplotlib.pyplot as plt
import numpy as np
import pygame
import tensorflow as tf
from tensorflow import keras

from cmdargs import args

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class RLModel:
    def __init__(self, env: gym.Env, state_shape: int, action_shape: int, 
                 mode: str, difficulty: int, train_episodes: int, fast: bool, 
                 render_fps: int, seed: int | None = None) -> None:
        self.env = env
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.mode = mode
        self.difficulty = difficulty
        self.train_episodes = train_episodes
        self.fast = fast
        self.render_fps = render_fps
        self.seed = seed

        # Initialize variables that determine the behaviour of 
        # the searching of the action space
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.03

        self.time_intvl_factor = 1

        self.train_target_steps = 15 if not self.fast else 30
        self.update_target_stesp = 400 if not self.fast else 600
        self.save_model_steps = 10 if not self.fast else 5
        
        # Maximum time elapsed (in minute) in fast mode
        self.fast_minute = 0.5

        # Number of neurons for each layer
        self.neurons = (256, 128, 128, 64, 32) if not self.fast else (128, 64, 32)

    def run(self) -> None:
        self.rewards, self.epsilons, self.scores, self.steps = [], [], [], []

        # Initialize the two models
        self.main_model = self._agent(self.neurons)
        self.target_model = self._agent(self.neurons)

        # Copy main model's weights to target model
        self.target_model.set_weights(self.main_model.get_weights())

        # Print the summary of the target model
        print(self.target_model.summary())
        print("=" * 65)

        # Initialize a fixed-sized list to store states, actions and rewards
        self.replay_memory = deque(maxlen=20_000)

        steps_to_update_target_model = 0

        time_intvl = int(self.time_intvl_factor * self.render_fps)

        # Get the starting time of the training
        self.start_time = time()

        self.episode = 0
        running = True
        while self.episode < self.train_episodes and running:
            self.episode += 1
            total_training_rewards = 0

            # Reset the environment
            state, reset_info = self.env.reset()

            terminated, truncated = False, False
            reward_interval = deque(maxlen=time_intvl)
            reward_interval_shoot = deque(maxlen=time_intvl)
            step = 0
            while not (terminated or truncated):
                if not running:
                    break

                # Detect events and pressed keys for quitting the game
                if self.mode == "human":
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False

                    pressed_keys = pygame.key.get_pressed()
                    if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:
                        running = False

                steps_to_update_target_model += 1
                step += 1

                # Pick action either randomly or not
                random_num = np.random.rand()
                if random_num <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_reshaped = state.reshape((1, state.shape[0]))
                    predicted = self.main_model.predict(state_reshaped, verbose=0)
                    action = np.argmax(predicted)

                # Take action and get reward
                new_state, reward, terminated, truncated, info = self.env.step(action)

                reward_interval_shoot.append(0)
                reward_interval.append(reward)
                reward_mean = np.array(reward_interval).mean()
                self.replay_memory.append([state, action, reward, new_state, terminated])
                if info["bullet lifetime"] is not None:
                    if info["bullet lifetime"] <= time_intvl:
                        reward_interval_shoot[-info["bullet lifetime"]] += 500
                    else:
                        self.replay_memory[-info["bullet lifetime"]][2] += 500

                if step > time_intvl:
                    self.replay_memory[-time_intvl][2] = reward_mean + reward_interval_shoot[0]
                if steps_to_update_target_model % self.train_target_steps == 0 or (terminated or truncated):
                    self._train(terminated)

                state = new_state
                total_training_rewards += reward

                if terminated or truncated:
                    for i in range(1, min(time_intvl, step) + 1):
                        self.replay_memory[-i][2] = np.array(list(islice(reward_interval, i - 1, None))).mean() + \
                                                    reward_interval_shoot[i - 1]

                    self.rewards.append(total_training_rewards)
                    self.epsilons.append(self.epsilon)
                    self.scores.append(info['score'])
                    self.steps.append(info['steps'])

                    # Save the target model regularly
                    if self.episode % self.save_model_steps == 0 or self.episode == self.train_episodes:
                        self.save()

                    # Print progress wrt time
                    time_elapsed = self._timer(self.start_time, self.episode, self.train_episodes)

                    # Print episode's training result
                    print(f"Total training reward = {total_training_rewards:<9.2f} "
                          f"at episode {self.episode:<{len(str(self.train_episodes))}d} "
                          f"with score = {info['score']:<2d}, steps = {info['steps']}")

                    # Copy main model's weights to target model
                    if steps_to_update_target_model >= self.update_target_stesp:
                        self.target_model.set_weights(self.main_model.get_weights())
                        steps_to_update_target_model = 0

                    if time_elapsed >= self.fast_minute * 60 and self.fast:
                        running = False

            # Garbage collection for memory issue
            gc.collect()
            keras.backend.clear_session()

            print("Epsilon:", self.epsilon)
            print("=" * 65)

            # Update epsilon
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)

        self.env.close()

        print(f"Avg score: {self._average(self.scores):.2f}, Avg steps: {self._average(self.steps):.2f}")

    def _agent(self, neurons: tuple[int]):
        learning_rate = 0.001

        init = tf.keras.initializers.HeUniform(seed=self.seed)
        model = keras.Sequential()

        for i, neuron in enumerate(neurons):
            model.add(
                keras.layers.Dense(
                    neuron,
                    input_shape=self.state_shape,
                    activation='relu',
                    kernel_initializer=init,
                    name=f"dense_{i}",
                )
            )

        model.add(
            keras.layers.Dense(
                self.action_shape,
                activation='linear',
                kernel_initializer=init,
                name=f"dense_{len(neurons)}",
            )
        )

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy'],
        )

        return model

    def _train(self, terminated) -> None:
        learning_rate = 0.7
        discount_factor = 0.618
        batch_size = 512
        min_replay_size = 1_000

        if len(self.replay_memory) < min_replay_size:
            return

        sampled_batch = random.sample(self.replay_memory, batch_size)
        old_states = np.array([batch[0] for batch in sampled_batch])
        old_qs_list = self.main_model.predict(old_states, verbose=0)
        new_states = np.array([batch[3] for batch in sampled_batch])
        new_qs_list = self.target_model.predict(new_states, verbose=0)

        x, y = [], []
        for idx, (state, action, reward, new_state, terminated) in enumerate(sampled_batch):
            if not terminated:
                max_future_q = reward + discount_factor * np.max(new_qs_list[idx])
            else:
                max_future_q = reward

            current_qs = old_qs_list[idx]

            # Update Q-value
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            x.append(state)
            y.append(current_qs)

        self.main_model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0, shuffle=True)

    def save(self) -> None:
        """A method that save the target model."""

        self.target_model.save(f"models/model_{'fast_' if self.fast else ''}diff_{self.difficulty}_epi_{self.episode}_{strftime('%H-%M-%S', gmtime(time() - self.start_time))}.h5")

    def plot(self) -> None:
        """
        A method that plots rewards, scores, steps and epsilons wrt 
        training episodes.
        """

        fig = plt.figure(figsize=(10, 8))

        x = np.arange(1, self.episode + 1)

        # Plot rewards
        ax1 = fig.add_subplot(221)
        ax1.plot(x, self.rewards, "o")
        m, b = np.polyfit(x, self.rewards, 1)
        ax1.plot(x, m * x + b)
        ax1.set_title("Rewards over all episodes in training")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")

        # Plot scores
        ax2 = fig.add_subplot(222)
        ax2.plot(x, self.scores, "o")
        m, b = np.polyfit(x, self.scores, 1)
        ax2.plot(x, m * x + b)
        ax2.set_title(f"Scores over all episodes in training | Average: {self._average(self.scores):.2f}")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Scores")

        # Plot steps
        ax3 = fig.add_subplot(223)
        ax3.plot(x, self.steps, "o")
        m, b = np.polyfit(x, self.steps, 1)
        ax3.plot(x, m * x + b)
        ax3.set_title(f"Steps over all episodes in training | Average: {self._average(self.steps):.2f}")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Steps")

        # Plot epsilons
        ax4 = fig.add_subplot(224)
        ax4.plot(x, self.epsilons)
        ax4.set_title("Epsilons over all episodes in training")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")

        # Adjust the padding between subplots
        fig.tight_layout()

        # Save the figure
        plt.savefig(f"training_results/training_result_{datetime.now().strftime('%H%M%S')}.png", dpi=300)

        # Show the figure
        plt.show()

    @staticmethod
    def _timer(start_time, progress, total) -> float:
        # Code source: https://github.com/dmddjack/ESTR2018_Project/blob/main/wordle_bot.py
        """
        An internal function that prints the current progress, time elapsed 
        and Estimated remaining time. Modified from FONG Shi Yuk's ESTR 2018 Project.
        """

        time_elapsed = time() - start_time

        print(f"Progress: {progress}/{total} ({progress / total * 100:.2f}%)")
        print(f"Time elapsed: {strftime('%H:%M:%S', gmtime(time_elapsed))}")
        print(f"Estimated time remaining: "
              f"{strftime(f'%H:%M:%S', gmtime(time_elapsed / (progress / total) - time_elapsed))}")

        return time_elapsed

    @staticmethod
    def _average(lst: list[int | float]) -> float:
        """An internal function that returns the average of a list of numbers"""

        return sum(lst) / len(lst)


def main():
    # Make a directory to store target models if necessary
    if not os.path.isdir("models"):
        os.mkdir("models")

    # Make a directory to store training results if necessary
    if not os.path.isdir("training_results"):
        os.mkdir("training_results")

    assert args.mode != "human_rand"

    env = gym.make(
        "gym_tankwar/TankWar-v0",
        render_mode=args.mode,
        starting_hp=args.starting_hp,
        difficulty=args.difficulty,
        episodes=args.episodes,
        full_enemy=args.full_enemy,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)

    env.action_space.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    observation_space_shape = env.observation_space.shape
    action_space_size = env.action_space.n

    print("Shape of observation space:", observation_space_shape)
    print("Size of action space      :", action_space_size)

    my_model = RLModel(
        env,
        observation_space_shape,
        action_space_size,
        args.mode,
        args.difficulty,
        args.train_episodes,
        args.fast,
        args.fps,
        args.seed,
    )
    my_model.run()
    my_model.plot()


if __name__ == "__main__":
    main()
