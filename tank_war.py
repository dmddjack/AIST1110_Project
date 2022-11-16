import random

import numpy as np
import pygame
from gym import Env, spaces

from assets import Enemy, Player


class TankWar(Env):
    metadata = {"render_modes": ["human", "read_only", "rgb_array"], "render_fps": 30}
    window_width = 600
    window_height = 400

    def __init__(self, render_mode: str, seed: None | int = None) -> None:
        super().__init__()
        self.rnd_seed = seed

        self.observation_shape = (self.window_height, self.window_width, 3)
        # self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
        #                                     high = np.ones(self.observation_shape))
        self.action_space = spaces.Discrete(10)

        # self.elements = []

        assert render_mode in self.metadata["render_modes"], "Invalid render mode, has to be \"human\", \"read_only\" or \"rgb_array\""
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self):
        # super().reset(seed=self.rnd_seed)
        self.steps = 0
        self.player = Player(self.window_width, self.window_height, 3, self.rnd_seed)
        self.enemy = Enemy(self.window_width, self.window_height, 2, self.rnd_seed)
        self.elements = [self.player, self.enemy]

        if self.render_mode == "human" or self.render_mode == "read_only":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Tank")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        # if self.render_mode.startswith("human"):
        #     self.render()

        # return obs

    def step(self, action):
        self.steps += 1

        d_x, d_y, angle = 0, 0, self.player.angle
        if action == 0:
            d_y = -1
            angle = 0
            print("up")
        if action == 1:
            d_y = 1
            angle = 180
            print("down")
        if action == 2:
            d_x = -1
            angle = 90
            print("left")
        if action == 3:
            d_x = 1
            angle = 270
            print("right")
        if action == 4:
            print("shoot")
        if action == 5:
            d_y = -1
            angle = 0
            print("up and shoot")
        if action == 6:
            d_y = 1
            angle = 180
            print("down and shoot")
        if action == 7:
            d_x = -1
            angle = 90
            print("left and shoot")
        if action == 8:
            d_x = 1
            angle = 270
            print("right and shoot")
        self.player.update(d_x, d_y, angle)

        d_x, d_y, angle = 0, 0, self.enemy.angle
        if self.steps != 0 and self.steps % (self.metadata["render_fps"] * 3) == 0:
            while angle == self.enemy.angle:
                angle = random.choice((0, 90, 180, 270))
            print("change")

        if angle == 0:
            d_y = -1
        elif angle == 90:
            d_x = -1
        elif angle == 180:
            d_y = 1
        else:
            d_x = 1
        self.enemy.update(d_x, d_y, angle)

        # return obs, reward, done, info

    def render(self):
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        
        bg = pygame.image.load("images/backgrounds/background_3.png").convert()
        bg.set_alpha(200)
        canvas.blit(bg, (0, 0))

        for element in self.elements:
            canvas.blit(element.surf, element.rect)

        if self.render_mode == "human" or self.render_mode == "read_only":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            # pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            
    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
