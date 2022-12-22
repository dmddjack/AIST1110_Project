#!/usr/bin/env python3

import gym
import gym_tankwar
import pygame

from cmdargs import args


def _pressed_to_action(pressed_keys, last_pressed_keys, last_action):
    """An internal function that maps pressed key(s) to an action"""

    def filter_dir(keys):
        dir_ = (keys[pygame.K_UP] or keys[pygame.K_w],    # Going up
               keys[pygame.K_DOWN] or keys[pygame.K_s],   # Going down
               keys[pygame.K_LEFT] or keys[pygame.K_a],   # Going left
               keys[pygame.K_RIGHT] or keys[pygame.K_d])  # Going right
        actions = []
        # if keys[4]:
        #     for i, value in enumerate(keys[:4]):
        #         if value:
        #             actions.append(i + 5)
        #     if not actions:
        #         actions.append(4)
        # else:
        for i, value in enumerate(dir_):
            if value:
                actions.append(i)
        if not actions:
            actions.append(4)  # stand still
        return actions

    if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:
        return None
    last_action_space = filter_dir(last_pressed_keys)
    action_space = filter_dir(pressed_keys)
    shoot = pressed_keys[pygame.K_SPACE]
    # print(shoot)
    if last_action is not None:
        # action = 4
        # if not pressed_keys[pygame.K_SPACE]:
        #     if pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_w]:
        #         action = 0
        #     if pressed_keys[pygame.K_DOWN] or pressed_keys[pygame.K_s]:
        #         action = 1
        #     if pressed_keys[pygame.K_LEFT] or pressed_keys[pygame.K_a]:
        #         action = 2
        #     if pressed_keys[pygame.K_RIGHT] or pressed_keys[pygame.K_d]:
        #         action = 3
        # else:
        #     action = 9
        #     if pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_w]:
        #         action = 5
        #     if pressed_keys[pygame.K_DOWN] or pressed_keys[pygame.K_s]:
        #         action = 6
        #     if pressed_keys[pygame.K_LEFT] or pressed_keys[pygame.K_a]:
        #         action = 7
        #     if pressed_keys[pygame.K_RIGHT] or pressed_keys[pygame.K_d]:
        #         action = 8
        # return action
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
    render_mode = args.mode
    if render_mode == "human_rand":
        render_mode = "human"

    env = gym.make(
        "gym_tankwar/TankWar-v0",
        render_mode=render_mode,
        starting_hp=args.starting_hp,
        difficulty=args.difficulty
    )

    if args.mode != "human":
        env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_steps)

    if args.fps is not None:
        env.metadata["render_fps"] = args.fps

    env.action_space.seed(args.seed)

    observation, info = env.reset(seed=args.seed)

    episode = 1
    success_episodes = 0
    running = True
    step = 0
    score = 0
    total_score = 0
    total_steps = 0
    action = None
    pressed_keys = None
    while running and episode <= args.episodes:
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
        else:
            action = env.action_space.sample()  # random

        if action is not None:
            observation, reward, terminated, truncated, info = env.step(action)

            step += 1
            score += reward

            if terminated or truncated:
                observation, info = env.reset(seed=args.seed)

                # Print the episode's final result
                if terminated:
                    print(
                        f"Episode {episode:<5d} " \
                        f"succeeded in {step:<5d} " \
                        f"steps ...\tReward = {score}"
                    )
                    success_episodes += 1
                else:
                    print(
                        f"Episode {episode:<5d} " \
                        f"truncated ...\t\t\tScore = {score}"
                    )

                episode += 1
                total_steps += step
                step = 0
                total_score += score
                score = 0

    if episode > 0:
        # Print success rate of all episodes
        print(
            f"Success rate = {success_episodes / episode:.2f}    " \
            f"Average steps = {total_steps // episode}    " \
            f"Average reward = {total_score / episode:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
