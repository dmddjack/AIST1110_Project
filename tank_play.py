import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from tank import Tank
import argparse


def pressed_to_action(pressed_keys):
    action = 9
    if pressed_keys[pygame.K_UP]:
        action = 0
    if pressed_keys[pygame.K_DOWN]:
        action = 1
    if pressed_keys[pygame.K_LEFT]:
        action = 2
    if pressed_keys[pygame.K_RIGHT]:
        action = 3
    if pressed_keys[pygame.K_SPACE]:
        action = 4
    if pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_SPACE]:
        action = 5
    if pressed_keys[pygame.K_DOWN] and pressed_keys[pygame.K_SPACE]:
        action = 6
    if pressed_keys[pygame.K_LEFT] and pressed_keys[pygame.K_SPACE]:
        action = 7
    if pressed_keys[pygame.K_RIGHT] and pressed_keys[pygame.K_SPACE]:
        action = 8
    # if pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_LEFT]:
    #     action = 9
    # if pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_RIGHT]:
    #     action = 10
    # if pressed_keys[pygame.K_DOWN] and pressed_keys[pygame.K_LEFT]:
    #     action = 11
    # if pressed_keys[pygame.K_DOWN] and pressed_keys[pygame.K_RIGHT]:
    #     action = 12
    # if pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_LEFT] and pressed_keys[pygame.K_SPACE]:
    #     action = 13
    # if pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_RIGHT] and pressed_keys[pygame.K_SPACE]:
    #     action = 14
    # if pressed_keys[pygame.K_DOWN] and pressed_keys[pygame.K_LEFT] and pressed_keys[pygame.K_SPACE]:
    #     action = 15
    # if pressed_keys[pygame.K_DOWN] and pressed_keys[pygame.K_RIGHT] and pressed_keys[pygame.K_SPACE]:
    #     action = 16
    if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:
        action = -1
    return action

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    args = parser.parse_args()
    render_mode = str(args.mode)
    try:
        seed = None if args.seed is None else int(args.seed)
        episodes = int(args.episodes)
        max_steps = int(args.max_steps)
    except ValueError:
        print("Invalid argument(s).")

    
    env = Tank(render_mode=render_mode)
    for episode in range(1, episodes + 1):
        env.reset(seed)
        if render_mode == "human":
            running = True
            while running:
                # env.clock.tick(env.metadata["render_fps"])
                for event in pygame.event.get():
                    # print(event)
                    if event.type == pygame.QUIT:
                        running = False
                pressed_keys = pygame.key.get_pressed()
                action = pressed_to_action(pressed_keys)
                if action == -1:
                    running = False
                else:
                    env.step(action)
                    env.render()
        elif render_mode == "read_only":
            while True:
                action = env.action_space.sample()
                # obs, reward, done, info = env.step(action)
                env.step(action)
                env.render()

                # if done == True:
                #     break
    