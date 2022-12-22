# Code source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import gym
import gym_tankwar
import numpy as np
from tensorflow import keras

from cmdargs import args


def main():
    assert args.mode != "human_rand"

    env = gym.make(
        "gym_tankwar/TankWar-v0", 
        render_mode=args.mode, 
        starting_hp=args.starting_hp,
        difficulty=args.difficulty,
    )

    if args.mode != "human":
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)

    env.action_space.seed(args.seed)

    model = keras.models.load_model(f"models/{args.file}.h5")

    print("Testing started ...")
    success_episodes = 0
    for episode in range(args.test_episodes):
        state, info = env.reset()
        total_testing_rewards = 0
        for step in range(args.max_steps):
            predicted = model.predict(state.reshape(1, state.shape[0]), verbose=0)
            action = np.argmax(predicted)
            state, reward, terminated, truncated, info = env.step(action) # take action and get reward
            total_testing_rewards += reward    
            if terminated or truncated:  # End the episode
                print(f"Episode {episode} succeeded in {step + 1} steps ...")
                success_episodes += 1
                break
        else:
            print(f"Episode {episode} truncated ...")

    print(f"Success rate: {success_episodes/args.test_episodes:.2f}")

    env.close()


if __name__ == "__main__":
    main()
