import gym
import gym_tankwar
import pygame

from cmdargs import args


def _pressed_to_action(pressed_keys):
    """An internal function that maps pressed key(s) to an action"""

    if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:
        return None
    action = 9
    if not pressed_keys[pygame.K_SPACE]:
        if pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_w]:
            action = 0
        if pressed_keys[pygame.K_DOWN] or pressed_keys[pygame.K_s]:
            action = 1
        if pressed_keys[pygame.K_LEFT] or pressed_keys[pygame.K_a]:
            action = 2
        if pressed_keys[pygame.K_RIGHT] or pressed_keys[pygame.K_d]:
            action = 3
    else:
        action = 4
        if pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_w]:
            action = 5
        if pressed_keys[pygame.K_DOWN] or pressed_keys[pygame.K_s]:
            action = 6
        if pressed_keys[pygame.K_LEFT] or pressed_keys[pygame.K_a]:
            action = 7
        if pressed_keys[pygame.K_RIGHT] or pressed_keys[pygame.K_d]:
            action = 8
    return action


def main():
    render_mode = args.mode
    if render_mode == "human_rand":
        render_mode = "human"

    episodes = args.episodes
    max_steps = args.max_steps

    env = gym.make(
        "gym_tankwar/TankWar-v0", render_mode=render_mode, starting_hp=args.starting_hp
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    if args.fps is not None:
        env.metadata["render_fps"] = args.fps

    env.action_space.seed(args.seed)

    observation, info = env.reset(seed=args.seed)

    episode = 0
    success_episodes = 0
    running = True
    step = 0
    score = 0
    total_score = 0

    while running and episode < episodes:
        if args.mode == "human":
            # Detect pygame events for quiting the game
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Detect pressed keys
            pressed_keys = pygame.key.get_pressed()

            # Map pressed keys to an action
            action = _pressed_to_action(pressed_keys)
            if action == None:
                running = False

        else:
            action = env.action_space.sample()  # random

        if action is not None:
            observation, reward, done, truncated, info = env.step(action)
            # print(episode, action, observation, reward, info)
            step += 1
            score += reward
            if done or truncated:
                observation, info = env.reset(seed=args.seed)
                if done:
                    print(
                        f"Episode {episode:<{len(str(episodes))}d} succeeded in {step:<{len(str(max_steps))}d} steps ...\tScore = {score}"
                    )
                    success_episodes += 1
                else:
                    print(
                        f"Episode {episode:<{len(str(episodes))}d} truncated ...\t\t\tScore = {score}"
                    )
                episode += 1
                step = 0
                total_score += score
                score = 0

    if episode > 0:
        print(
            f"Success rate = {success_episodes/episode:.2f}    Average score = {total_score/episode:.2f}"
        )

    env.close()


if __name__ == "__main__":
    main()
