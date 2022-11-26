# Code source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import gym
import gym_tankwar
import numpy as np
from tensorflow import keras

from cmdargs import args


def main():
    env = gym.make(
        "gym_tankwar/TankWar-v0", 
        render_mode=args.mode, 
        starting_hp=args.starting_hp, 
        window_width=args.window_width, 
        window_height=args.window_height, 
        max_enemies=args.max_enemies,
    )
    env.action_space.seed(args.seed)

    model = keras.models.load_model(f"models/model.h5")

    test_episodes = 1000
    max_steps = 5400

    print("Testing started ...")
    success_episodes = 0
    for episode in range(test_episodes):
        state, info = env.reset()
        total_testing_rewards = 0
        for step in range(max_steps):
            predicted = model.predict(state.reshape(1, state.shape[0]), verbose=0)
            action = np.argmax(predicted)
            state, reward, terminated, truncated, info = env.step(action) # take action and get reward
            total_testing_rewards += reward    
            if terminated or total_testing_rewards >= 15: # End the episode
                print(f"Episode {episode} succeeded in {step + 1} steps ...")
                success_episodes += 1
                break
        else:
            print(f"Episode {episode} truncated ...")

    print(f"Success rate: {success_episodes/test_episodes:.2f}")

    env.close()


if __name__ == "__main__":
    main()
