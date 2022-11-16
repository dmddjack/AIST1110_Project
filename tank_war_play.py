import os

# Disable pygame welcome message (https://stackoverflow.com/questions/51464455/how-to-disable-welcome-message-when-importing-pygame)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import argparse

import pygame

from tank_war import TankWar


def pressed_to_action(pressed_keys):
    if pressed_keys[pygame.K_q] or pressed_keys[pygame.K_ESCAPE]:
        return -1
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, required=True, help="\"human\", \"read_only\" or \"rgb_array\"")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-e", "--episodes", type=int, required=True)
    parser.add_argument("-ms", "--max_steps", type=int, default=3600)
    args = parser.parse_args()
    render_mode = str(args.mode)
    try:
        seed = None if args.seed is None else int(args.seed)
        episodes = int(args.episodes)
        max_steps = int(args.max_steps)
    except ValueError:
        print("Invalid argument(s).")

    env = TankWar(render_mode, seed, max_steps)
    for episode in range(1, episodes + 1):
        env.reset()
        if render_mode == "human":
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                pressed_keys = pygame.key.get_pressed()
                action = pressed_to_action(pressed_keys)
                if action == -1:
                    running = False
                else:
                    terminated, info = env.step(action)
                    env.render()

                    if terminated:
                        running = False
        elif render_mode == "read_only":
            while True:
                action = env.action_space.sample()
                # obs, reward, done, info = env.step(action)
                terminated, info = env.step(action)
                env.render()

                if terminated:
                    break
    env.close()
    
