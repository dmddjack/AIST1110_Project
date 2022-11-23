#!/usr/bin/env python3

import gym
import gym_tankwar
import pygame
from gym.utils import play

from cmdargs import args

# Play the game in human-controlled mode


def main():
    env = gym.make(
        "gym_tankwar/TankWar-v0", 
        render_mode="rgb_array", 
        starting_hp=args.starting_hp
    )

    mapping = {
        (pygame.K_UP,): 0,
        (pygame.K_w,): 0,
        (pygame.K_DOWN,): 1,
        (pygame.K_s,): 1,
        (pygame.K_LEFT,): 2,
        (pygame.K_a,): 2,
        (pygame.K_RIGHT,): 3,
        (pygame.K_d,): 3,
        (pygame.K_SPACE,): 4,
        (pygame.K_UP, pygame.K_SPACE): 5,
        (pygame.K_w, pygame.K_SPACE): 5,
        (pygame.K_DOWN, pygame.K_SPACE): 6,
        (pygame.K_s, pygame.K_SPACE): 6,
        (pygame.K_LEFT, pygame.K_SPACE): 7,
        (pygame.K_a, pygame.K_SPACE): 7,
        (pygame.K_RIGHT, pygame.K_SPACE): 8,
        (pygame.K_d, pygame.K_SPACE): 8,
        (pygame.NOEVENT,): 9,
    }

    play.play(
        env=env, 
        fps=args.fps, 
        keys_to_action=mapping, 
        seed=args.seed, 
        noop=None
    )


if __name__ == "__main__":
    main()
