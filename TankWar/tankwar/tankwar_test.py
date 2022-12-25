#!/usr/bin/env python3

# Code source: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import gym
import gym_tankwar
import numpy as np
import pygame
from tensorflow import keras

from cmdargs import args


def main():
    assert args.mode != "human_rand"

    env = gym.make(
        "gym_tankwar/TankWar-v0",
        render_mode=args.mode,
        starting_hp=args.starting_hp,
        difficulty=args.difficulty,
        full_enemy=args.full_enemy,
    )

    env.action_space.seed(args.seed)

    model = keras.models.load_model(f"models/{args.file}.h5")

    print("Testing started ...")
    success_episodes = 0
    total_score = total_step = 0
    episode = 0
    running = True
    while running and episode < args.test_episodes:
        episode += 1

        state, info = env.reset()
        total_testing_rewards = 0
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

            state, reward, terminated, truncated, info = env.step(action)  # take action and get reward
            total_testing_rewards += reward

            if terminated or total_testing_rewards >= 50000:  # End the episode
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
