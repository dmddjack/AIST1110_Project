#!/usr/bin/env python3

# Code source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Use GPU acceleration if possible

import gym
import gym_tankwar
import numpy as np
import pygame
from tensorflow import keras

from cmdargs import args


def main():
    assert args.mode != "human_rand", "human_rand mode cannot be used here"
    assert args.test_episodes > 0, "TEST_EPISODES must be a positive integer"
    assert args.max_steps > 0, "MAX_STEPS must be a positive integer"
    assert args.file is not None, "FILE cannot be None"

    env = gym.make(
        "gym_tankwar/TankWar-v0",
        render_mode=args.mode,
        starting_hp=args.starting_hp,
        difficulty=args.difficulty,
        episodes=args.test_episodes,
        full_enemy=args.full_enemy,
    )

    env.action_space.seed(args.seed)

    # Load the model
    model = keras.models.load_model(f"models/{args.file}.h5", compile=False)

    print("Testing started ...")
    episode = success_episodes = 0
    total_score = total_step = 0
    running = True
    while running and episode < args.test_episodes:
        episode += 1
        total_testing_rewards = 0

        # Reset the environment
        state, reset_info = env.reset()

        for step in range(1, args.max_steps + 1):
            if not running:
                break

            # Detect events and pressed keys for quitting the game
            if args.mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                pressed_keys = pygame.key.get_pressed()
                if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:
                    running = False

            # Get action from the model
            predicted = model.predict(state.reshape(1, state.shape[0]), verbose=0)
            action = np.argmax(predicted)

            # Take action and get reward
            state, reward, terminated, truncated, info = env.step(action)
            total_testing_rewards += reward

            # End the episode
            if terminated or total_testing_rewards >= 50000:
                success_episodes += 1
                total_score += info["score"]
                total_step += step
                print(f"Episode {episode:<{len(str(args.test_episodes))}d} "
                      f"completed in {step:<{len(str(args.max_steps))}d} "
                      f"steps with score = {info['score']}")
                break

        else:
            print(f"Episode {episode} truncated ...")

    print(f"Completion rate: {success_episodes / episode:.2f}")
    print(f"Avg score: {total_score / success_episodes:.2f}, Avg steps: {total_step / success_episodes:.2f}")

    env.close()


if __name__ == "__main__":
    main()
