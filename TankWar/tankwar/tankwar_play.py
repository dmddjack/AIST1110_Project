#!/usr/bin/env python3

import gym
import gym_tankwar
import pygame
import random

from cmdargs import args


def _pressed_to_action(pressed_keys, last_pressed_keys, last_action) -> int | None:
    """An internal function that maps pressed key(s) to an action"""

    def filter_dir(keys) -> list[int]:
        """An internal function maps pressed key(s) to a list of actions"""
        dir_ = (keys[pygame.K_UP] or keys[pygame.K_w],  # Going up
                keys[pygame.K_DOWN] or keys[pygame.K_s],  # Going down
                keys[pygame.K_LEFT] or keys[pygame.K_a],  # Going left
                keys[pygame.K_RIGHT] or keys[pygame.K_d])  # Going right

        actions = []
        for i, value in enumerate(dir_):
            if value:
                actions.append(i)

        if not actions:
            actions.append(4)  # Stand still

        return actions

    if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:  # Keys for quitting the game
        return None
    if pressed_keys[pygame.K_r]:  # Key for restarting the game
        return -100
    if pressed_keys[pygame.K_RETURN]:  # Key for starting the game
        return -200

    last_action_space = filter_dir(last_pressed_keys)
    action_space = filter_dir(pressed_keys)
    shoot = pressed_keys[pygame.K_SPACE]

    if last_action is not None:
        last_action %= 5
        if last_action_space == action_space and last_action in action_space:
            return last_action + 5 * shoot
        elif last_action in action_space and len(action_space) > 1:
            tmp = action_space[:]
            for action in tmp:
                if action in last_action_space:
                    action_space.remove(action)
            if not action_space:
                return last_action + 5 * shoot

    return action_space[0] + 5 * shoot


def main():
    assert args.episodes > 0, "EPISODES must be a positive integer"
    assert args.max_steps > 0, "MAX_STEPS must be a positive integer"

    render_mode = args.mode
    if render_mode == "human_rand":
        render_mode = "human"

    env = gym.make(
        "gym_tankwar/TankWar-v0",
        render_mode=render_mode,
        starting_hp=args.starting_hp,
        difficulty=args.difficulty,
        full_enemy=args.full_enemy,
        episodes=args.episodes,
        extra_scene=True if args.mode == "human" else False,
    )

    # The extra +1 is for the last game-over scene in human mode
    episodes = args.episodes + 1 if args.mode == "human" else args.episodes


    if args.mode != "human":
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)

    if args.fps is not None:
        env.metadata["render_fps"] = args.fps

    env.action_space.seed(args.seed)
    random.seed(args.seed)

    # Use random.randint to generate a sequence of seeds from args.seed
    # to match the same implementation in tankwar_test.py
    observation, reset_info = env.reset(seed=random.randint(0, 2 ** 32 - 1))
    
    episode = 1
    success_episodes = 0
    running = True
    rewards = step = 0
    total_rewards = total_score = total_steps = 0
    action = None
    pressed_keys = None
    beginning, gameover = True, False
    while running and episode < episodes + 1:
        if args.mode == "human":
            # Detect pygame events for quiting the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Detect pressed keys
            last_action = action

            if pressed_keys is not None:
                last_pressed_keys = pressed_keys
            else:
                last_pressed_keys = pygame.key.get_pressed()

            pressed_keys = pygame.key.get_pressed()

            # Map pressed keys to an action
            action = _pressed_to_action(pressed_keys, last_pressed_keys, last_action)
            if action is None:
                running = False

            # Check if the game just starts
            if beginning:
                if action == -200:
                    beginning = False
                else:
                    continue
            else:  # Check if the Enter key is pressed when the game is not over
                if action == -200:
                    action = 4

            # Check if the game is over
            if gameover:
                if action == -100:  # Check if the R key is pressed when the game is over
                    if episode != episodes:
                        gameover = False
                        # observation, reset_info = env.reset(seed=args.seed)
                        observation, reset_info = env.reset(seed=random.randint(0, 2 ** 32 - 1))
                    else:
                        continue
                else:
                    continue
            else:  # Check if the R key or Enter key is pressed when the game is not over
                if action == -200:
                    action = 4
        else:
            # Detect events and pressed keys for quitting the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:
                running = False

            # Pick an action from the action space randomly
            action = env.action_space.sample()

            # Check if the game is over
            if gameover:
                gameover = False
                # observation, reset_info = env.reset(seed=args.seed)
                observation, reset_info = env.reset(seed=random.randint(0, 2 ** 32 - 1))

        if action is not None:
            # Take action and get reward
            observation, reward, terminated, truncated, info = env.step(action)

            step += 1
            rewards += reward

            if terminated or truncated:
                # Print episode's final result
                if terminated:
                    print(
                        f"Episode {episode:<{len(str(args.episodes))}d} "
                        f"completed in {step:<{len(str(args.max_steps))}d} "
                        f"steps with reward = {rewards:<9.2f}, "
                        f"score = {info['score']}"
                    )
                    success_episodes += 1
                else:
                    print(
                        f"Episode {episode:<{len(str(args.episodes))}d} "
                        f"truncated with reward = {rewards:<9.2f}, "
                        f"score = {info['score']}"
                    )

                # Keep counting variables
                episode += 1
                total_steps += step
                total_score += info["score"]
                total_rewards += rewards

                # Reset variables
                step = 0
                rewards = 0

                gameover = True

    # Print overall results
    print(
        f"Completion rate = {success_episodes/(episode-1) if episode > 1 else 0:.2f}, "
        f"Avg score = {total_score/success_episodes if success_episodes > 0 else 0:.2f}, "
        f"Avg steps = {total_steps/success_episodes if success_episodes > 0 else 0:.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()
