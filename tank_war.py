import random

import numpy as np
import pygame
from gym import Env, spaces

from assets import Enemy, EnemyBullet, Player, PlayerBullet


class TankWar(Env):
    metadata = {"render_modes": ["human", "read_only", "rgb_array"], "render_fps": 30}
    window_width = 600
    window_height = 400

    def __init__(self, render_mode: str, seed: None | int = None, max_steps: int = 3600) -> None:
        super().__init__()
        self.rnd_seed = seed
        self.max_steps = max_steps

        self.observation_shape = (self.window_height, self.window_width, 3)
        self.observation_space = spaces.Box(0, max(self.window_width, self.window_height), shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)

        assert render_mode in self.metadata["render_modes"], "Invalid render mode, has to be \"human\", \"read_only\" or \"rgb_array\""
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        player_dx, player_dy, player_angle = 0, 0, self.player.angle
        if action == 0 or action == 5:
            player_dy = -1
            player_angle = 0
            if action == 5:
                print("up and shoot")
        elif action == 1 or action == 6:
            player_dy = 1
            player_angle = 180
            if action == 6:
                print("down and shoot")
        elif action == 2 or action == 7:
            player_dx = -1
            player_angle = 90
            if action == 7:
                print("left and shoot")
        elif action == 3 or action == 8:
            player_dx = 1
            player_angle = 270
            if action == 8:
                print("right and shoot")
        elif action == 4:
            print("shoot")
        self.player.update(player_dx, player_dy, player_angle)

        for enemy in self.enemies:
            enemy_dx, enemy_dy, enemy_angle = 0, 0, enemy.angle
            if self.steps != 0 and self.steps % (self.metadata["render_fps"] * 3) == 0:
                while enemy_angle == enemy.angle:
                    enemy_angle = random.choice((0, 90, 180, 270))

            if enemy_angle == 0:
                enemy_dy = -1
            elif enemy_angle == 90:
                enemy_dx = -1
            elif enemy_angle == 180:
                enemy_dy = 1
            else:
                enemy_dx = 1
            enemy.update(enemy_dx, enemy_dy, enemy_angle)

        if pygame.sprite.spritecollideany(self.player, self.enemies):
            # The player will not disappear iff the following line is commented
            # self.player.kill()
            terminated = True

        if self.steps == self.max_steps:
            terminated = True

        # Create a placeholder for info
        info = {}
        # return obs, reward, done, info
        return reward, terminated, info

    def reset(self):
        super().reset(seed=self.rnd_seed)
        self.steps = 0
        self.score = 0

        self.all_sprites = pygame.sprite.Group()
        self.player = Player(self.window_width, self.window_height, 3, self.rnd_seed)
        self.all_sprites.add(self.player)
        self.enemies = pygame.sprite.Group()
        for _ in range(4):
            new_enemy = Enemy(self.window_width, self.window_height, 2, self.rnd_seed)
            self.enemies.add(new_enemy)
            self.all_sprites.add(new_enemy)

        if self.render_mode == "human" or self.render_mode == "read_only":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Tank War")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        # return obs

    def render(self):
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        
        bg = pygame.image.load("images/backgrounds/background_3.png").convert()
        bg.set_alpha(200)
        canvas.blit(bg, (0, 0))

        for sprite in self.all_sprites:
            canvas.blit(sprite.surf, sprite.rect)

        if self.render_mode == "human" or self.render_mode == "read_only":
            self.window.blit(canvas, canvas.get_rect())
            # pygame.event.pump()
            pygame.display.update()

            # Ensure the rendering occurs at the predefined framerate
            self.clock.tick(self.metadata["render_fps"])
        else: # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            
    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
