# Code source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random
from collections import deque

import gym
import gym_tankwar
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
from cmdargs import args


class RLModel:
    def __init__(self, env: gym.Env, state_shape, action_shape, train_episodes: int, seed: int | None = None):
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.02

        self.env = env
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.train_episodes = train_episodes
        self.seed = seed

    def run(self):
        self.rewards, self.epsilons = [], []

        self.main_model = self._agent()
        self.target_model = self._agent()
        self.target_model.set_weights(self.main_model.get_weights())

        print(self.target_model.summary())

        self.replay_memory = deque(maxlen=100_000)

        steps_to_update_target_model = 0

        for episode in range(1, 1 + self.train_episodes):
            total_training_rewards = 0
            state, info = self.env.reset()

            terminated, truncated = False, False
            while not (terminated or truncated):
                steps_to_update_target_model += 1

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
                self.replay_memory.append([state, action, reward, new_state, terminated])
                if steps_to_update_target_model % 30 == 0 or (terminated or truncated):
                    self._train(terminated)

                state = new_state
                total_training_rewards += reward

                if terminated or truncated:
                    self.rewards.append(total_training_rewards)
                    self.epsilons.append(self.epsilon)

                    if episode % 10 == 0:
                        self.save(episode)
                    print(f"Total training rewards = {total_training_rewards:<8.1f} at episode {episode:<4d} with score = {info['score']}, steps = {info['steps']}")

                    if steps_to_update_target_model >= 500:
                        self.target_model.set_weights(self.main_model.get_weights())
                        steps_to_update_target_model = 0

                    break
            gc.collect()
            keras.backend.clear_session()

            print("Epsilon:", self.epsilon)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * episode)

        self.env.close()

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
                128,
                activation='relu',
                kernel_initializer=init
            )
        )
        model.add(
            keras.layers.Dense(
                128,
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
        learning_rate = 0.5
        discount_factor = 0.618
        batch_size = 128
        min_replay_size = 1_000

        if len(self.replay_memory) < min_replay_size:
            return

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
        ax1.set_xlabel("Episode")
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
