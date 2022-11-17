import numpy as np
import pygame
from gym import Env, spaces

from assets import Enemy, Player


class TankWar(Env):
    metadata = {"render_modes": ["human", "read_only", "rgb_array"], "render_fps": 30}
    window_width = 600
    window_height = 400

    def __init__(self, render_mode: str, max_steps: int = 3600) -> None:
        super().__init__()

        self.max_steps = max_steps

        self.observation_shape = (self.window_height, self.window_width, 3)
        self.observation_space = spaces.Box(0, max(self.window_width, self.window_height), shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(10)

        assert render_mode in self.metadata["render_modes"], f"Invalid render mode, has to be {', '.join(self.metadata['render_modes'])}"
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

        # Move the player
        player_dx, player_dy, player_angle = 0, 0, self.player.angle
        if action == 0 or action == 5:
            player_dy = -1
            player_angle = 0
            if action == 5:
                self._player_shoot(player_angle)
        elif action == 1 or action == 6:
            player_dy = 1
            player_angle = 180
            if action == 6:
                self._player_shoot(player_angle)
        elif action == 2 or action == 7:
            player_dx = -1
            player_angle = 90
            if action == 7:
                self._player_shoot(player_angle)
        elif action == 3 or action == 8:
            player_dx = 1
            player_angle = 270
            if action == 8:
                self._player_shoot(player_angle)
        elif action == 4:
            self._player_shoot(player_angle)
        self.player.update(player_dx, player_dy, player_angle)

        # Create sufficient enemies
        self._create_enemy()

        ##### Better intelligence of the enemies is required
        # Move the enemies randomly
        for enemy in self.enemies:
            enemy_dx, enemy_dy, enemy_angle = 0, 0, enemy.angle
            if self.steps != 0 and self.steps % (self.metadata["render_fps"] * 3) == 0:
                while enemy_angle == enemy.angle:
                    enemy_angle = self.np_random.choice((0, 90, 180, 270))
            if enemy_angle == 0:
                enemy_dy = -1
            elif enemy_angle == 90:
                enemy_dx = -1
            elif enemy_angle == 180:
                enemy_dy = 1
            else:
                enemy_dx = 1
            enemy.update(enemy_dx, enemy_dy, enemy_angle)

        # Move the player's bullets
        for bullet in self.player_bullets:
            bullet.move()
            if bullet.rect.right < 0 or bullet.rect.left > self.window_width or bullet.rect.bottom <= 0 or bullet.rect.top >= self.window_height:
                bullet.kill()

        # Terminate the episode if the player has collided into any of the enemies
        if pygame.sprite.spritecollideany(self.player, self.enemies):
            # Remove the player from the window
            # self.player.kill()
            terminated = True

        for bullet in self.player_bullets:
            if len(pygame.sprite.spritecollide(bullet, self.enemies, dokill=True)):
                self.score += 10

                # Remove the bullet if it is outside the window
                bullet.kill()

        # Terminate the episode if the number of maximum steps is reached
        if self.steps == self.max_steps:
            terminated = True

        # Create a placeholder for info
        info = {}
        # return obs, reward, done, info
        return reward, terminated, info

    def _player_shoot(self, angle: int) -> None:
        if self.player_last_shot == 0 or self.steps - self.player_last_shot >= self.metadata["render_fps"]:
            self.player_last_shot = self.steps
            player_bullet = self.player.bullet(self.player.surf.get_size(), self.player.rect.center, angle, self.player.speed + 3)
            self.player_bullets.add(player_bullet)
            self.all_sprites.add(player_bullet)

    def reset(self, seed: None | int = None):
        # Seed the random number generator
        super().reset(seed=seed)

        # Reset all counting variables
        self.steps = 0
        self.score = 0
        self.player_last_shot = 0

        # Create a placeholder for all sprites
        self.all_sprites = pygame.sprite.Group()

        # Create the player
        player_start_x = int(self.np_random.integers(self.window_width * 0.2, self.window_width * 0.8, size=1))
        player_start_y = int(self.np_random.integers(self.window_height * 0.2, self.window_height * 0.8, size=1))
        player_start_angle = self.np_random.choice((0, 90, 180, 270))
        self.player = Player(player_start_x, player_start_y, player_start_angle, self.window_width, self.window_height, 4)
        self.all_sprites.add(self.player)

        # Create a placeholder for all enemies
        self.enemies = pygame.sprite.Group()

        # Create one enemy
        self._create_enemy()

        # Create placeholders for the player's and enemies' bullets
        self.player_bullets = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()

        # Initialize pygame if necessary
        if self.render_mode == "human" or self.render_mode == "read_only":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Tank War")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Garamond', 25)

        # return obs

    def _score2enemy(self) -> tuple[int, int]:
        enemy_n, enemy_speed = 0, 0
        if self.score < 50:
            enemy_n, enemy_speed = 1, 2
        elif self.score < 100:
            enemy_n, enemy_speed = 2, 2
        else:
            enemy_n, enemy_speed = 3, 3
        return enemy_n, enemy_speed

    def _create_enemy(self):
        enemy_n, enemy_speed = self._score2enemy()
        for _ in range(enemy_n - len(self.enemies)):
            enemy_start_angle = self.np_random.choice((0, 90, 180, 270))
            if enemy_start_angle == 0:
                enemy_start_x = int(self.np_random.integers(0, self.window_width, size=1))
                enemy_start_y = self.window_height
            elif enemy_start_angle == 90:
                enemy_start_x = self.window_width
                enemy_start_y = int(self.np_random.integers(0, self.window_height, size=1))
            elif enemy_start_angle == 180:
                enemy_start_x = int(self.np_random.integers(0, self.window_width, size=1))
                enemy_start_y = 0
            else:
                enemy_start_x = 0
                enemy_start_y = int(self.np_random.integers(0, self.window_height, size=1))
            enemy = Enemy(enemy_start_x, enemy_start_y, enemy_start_angle, self.window_width, self.window_height, enemy_speed)
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)

    def render(self) -> None | np.ndarray:
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        
        bg = pygame.image.load("images/backgrounds/background_3.png").convert()
        bg.set_alpha(200)
        canvas.blit(bg, (0, 0))

        for sprite in self.all_sprites:
            canvas.blit(sprite.surf, sprite.rect)

        score_surf = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        canvas.blit(score_surf, (5, 5))

        game_time_total = self.steps // self.metadata["render_fps"]
        game_time_min = game_time_total // 60
        game_time_sec = game_time_total - game_time_min * 60
        time_surf = self.font.render(f"Time: {game_time_min:0>2d}:{game_time_sec:0>2d}", True, (0, 0, 0))
        canvas.blit(time_surf, (5, 25))

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
