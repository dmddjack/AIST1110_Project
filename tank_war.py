import numpy as np
import pygame
from gym import Env, spaces

from assets import Enemy, Heart, Player


class TankWar(Env):
    metadata = {"render_modes": ("human", "read_only", "rgb_array"), "render_fps": 30}
    window_width, window_height = 600, 400
    player_speed = 4
    player_shoot_intvl = 1.0
    beginning_hp = 3

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
        player_shoot = False
        if action == 0 or action == 5: # Move up
            player_dy = -1
            player_angle = 0
            if action == 5: # Shoot while moving up
                player_shoot = True
        elif action == 1 or action == 6: # Move down
            player_dy = 1
            player_angle = 180
            if action == 6: # Shoot while moving down
                player_shoot = True
        elif action == 2 or action == 7: # Move left
            player_dx = -1
            player_angle = 90
            if action == 7: # Shoot while moving left
                player_shoot = True
        elif action == 3 or action == 8: # Move right
            player_dx = 1
            player_angle = 270
            if action == 8: # Shoot while moving right
                player_shoot = True
        elif action == 4: # Shoot while not moving
            player_shoot = True
        
        # Update the player's position
        self.player.update(player_dx, player_dy, player_angle)

        # Shoot a bullet from the player's position if player_shoot is true
        if player_shoot:
            self._player_shoot(player_angle)

        # Create sufficient enemies
        enemy_speed, enemy_shoot_intvl = self._create_enemy()

        # Move the enemies
        for enemy in self.enemies:
            enemy.speed = enemy_speed
            enemy_dx, enemy_dy, enemy_angle = 0, 0, enemy.angle

            # An enemy rotates with an interval of not less than 3 seconds and a probability of 1%
            if self.steps - enemy.last_rotate >= self.metadata["render_fps"] * 3 and self.np_random.random() < 0.01:
                while enemy_angle == enemy.angle:
                    enemy_angle = self.np_random.choice((0, 90, 180, 270))
                enemy.last_rotate = self.steps
            if enemy_angle == 0: # Move up
                enemy_dy = -1
            elif enemy_angle == 90: # Move down
                enemy_dx = -1
            elif enemy_angle == 180: # Move left
                enemy_dy = 1
            else: # Move right
                enemy_dx = 1
            
            # Update the enemy's position
            enemy.update(enemy_dx, enemy_dy, enemy_angle)

            # Shoot a bullet from the enemy's position
            self._enemy_shoot(enemy, enemy_angle, enemy_shoot_intvl)

        # Move the player's and enemies' bullets
        for bullets in (self.player_bullets, self.enemy_bullets):
            for bullet in bullets:
                bullet.move()

                # Remove the bullet if it is outside the window
                if bullet.rect.right < 0 or bullet.rect.left > self.window_width or bullet.rect.bottom <= 0 or bullet.rect.top >= self.window_height:
                    bullet.kill()

        # Terminate the episode if the player has collided into any of the enemies or any of the enemies' bullets
        if pygame.sprite.spritecollideany(self.player, self.enemies) or len(pygame.sprite.spritecollide(self.player, self.enemy_bullets, dokill=True)):
            self.hp -= 1
            reward -= 50
            if self.hp == 0:
                terminated = True

                # Remove the player
                self.player.kill()
            else:
                # Remove all enemies to ensure the player will not be killed at spawn
                for enemy in self.enemies:
                    enemy.kill()

                # Respawn the player at a randomly generated location
                player_respawn_x = int(self.np_random.integers(self.window_width * 0.3, self.window_width * 0.7, size=1))
                player_respawn_y = int(self.np_random.integers(self.window_height * 0.3, self.window_height * 0.7, size=1))
                player_respawn_angle = self.np_random.choice((0, 90, 180, 270))
                self.player.rect = self.player.surf.get_rect(center=(player_respawn_x, player_respawn_y))
                self.player.update(0, 0, player_respawn_angle)

            # Remove one heart
            self.hearts.sprites()[-1].kill()

        for bullet in self.player_bullets:
            # Remove the player's bullet if it hits an enemy
            if len(pygame.sprite.spritecollide(bullet, self.enemies, dokill=True)):
                bullet.kill()
                self.score += 10
                reward += 10

        for bullet in self.player_bullets:
            # Remove the player's bullet if it hits an enemy's bullet
            if len(pygame.sprite.spritecollide(bullet, self.enemy_bullets, dokill=True)):
                bullet.kill()

        # Terminate the episode if the number of maximum steps is reached
        if self.steps == self.max_steps:
            terminated = True

        # Create a placeholder for information
        info = {}
        # return obs, reward, done, info
        return reward, terminated, info

    def _player_shoot(self, angle: int) -> None:
        # The player can shoot with an interval of 1 second
        if self.player.last_shoot == 0 or self.steps - self.player.last_shoot >= self.metadata["render_fps"] * self.player_shoot_intvl:
            self.player.last_shoot = self.steps
            player_bullet = self.player.bullet(self.player.surf.get_size(), self.player.rect.center, angle, self.player.speed + 3)
            self.player_bullets.add(player_bullet)
            self.all_sprites.add(player_bullet)

    def _enemy_shoot(self, enemy: Enemy, angle: int, interval: int) -> None:
        # Enemies shoot with an interval of not less than 2 seconds and a probability of 5%
        if self.steps - enemy.last_shoot >= self.metadata["render_fps"] * interval and self.np_random.random() < 0.05:
            enemy.last_shoot = self.steps
            enemy_bullet = enemy.bullet(enemy.surf.get_size(), enemy.rect.center, angle, enemy.speed + 2)
            self.enemy_bullets.add(enemy_bullet)
            self.all_sprites.add(enemy_bullet)

    def reset(self, seed: None | int = None):
        # Seed the random number generator
        super().reset(seed=seed)

        # Reset all counting variables
        self.steps = 0
        self.score = 0
        self.hp = self.beginning_hp

        # Create a group for all sprites except hearts
        self.all_sprites = pygame.sprite.Group()

        # Create the player
        player_start_x = int(self.np_random.integers(self.window_width * 0.3, self.window_width * 0.7, size=1))
        player_start_y = int(self.np_random.integers(self.window_height * 0.3, self.window_height * 0.7, size=1))
        player_start_angle = self.np_random.choice((0, 90, 180, 270))
        self.player = Player(player_start_x, player_start_y, player_start_angle, self.window_width, self.window_height, self.player_speed)
        self.all_sprites.add(self.player)

        # Create a group for enemies
        self.enemies = pygame.sprite.Group()

        # Create one enemy
        self._create_enemy()

        # Create groups for the player's and enemies' bullets
        self.player_bullets = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()

        # Create a group for hearts
        self.hearts = pygame.sprite.Group()
        for i in range(1, self.beginning_hp + 1):
            heart = Heart(self.window_width, i)
            self.hearts.add(heart)
            self.all_sprites.add(heart)

        # Initialize pygame if necessary
        if self.render_mode == "human" or self.render_mode == "read_only":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Tank War")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Garamond', 25)

        # return obs

    def _score2enemy(self) -> tuple[int, int, float]:
        """An internal function that maps the current score to the behaviour of the enemies"""

        enemy_n, enemy_speed, enemy_shoot_intvl = 0, 0, 0
        if self.score < 50:
            enemy_n, enemy_speed, enemy_shoot_intvl = 1, 2, 2
        elif self.score < 100:
            enemy_n, enemy_speed, enemy_shoot_intvl = 1, 3, 2
        elif self.score < 150:
            enemy_n, enemy_speed, enemy_shoot_intvl = 2, 2, 2
        elif self.score < 200:
            enemy_n, enemy_speed, enemy_shoot_intvl = 2, 3, 2
        elif self.score < 250:
            enemy_n, enemy_speed, enemy_shoot_intvl = 3, 2, 1.5
        elif self.score < 300:
            enemy_n, enemy_speed, enemy_shoot_intvl = 3, 3, 1.5
        elif self.score < 350:
            enemy_n, enemy_speed, enemy_shoot_intvl = 4, 2, 1.5
        else:
            enemy_n, enemy_speed, enemy_shoot_intvl = 4, 3, 1.5
        return enemy_n, enemy_speed, enemy_shoot_intvl

    def _create_enemy(self) -> None:
        """An internal function that creates sufficient enemies"""

        enemy_n, enemy_speed, enemy_shooting_interval = self._score2enemy()
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
            enemy = Enemy(enemy_start_x, enemy_start_y, enemy_start_angle, self.window_width, self.window_height, enemy_speed, self.steps)
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)
        return enemy_speed, enemy_shooting_interval

    def render(self) -> None | np.ndarray:
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        
        bg = pygame.image.load("images/backgrounds/background_3.png").convert()
        bg.set_alpha(200)
        canvas.blit(bg, (0, 0))

        for sprite in self.all_sprites:
            canvas.blit(sprite.surf, sprite.rect)

        # Display the score on the window
        score_surf = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        canvas.blit(score_surf, (5, 5))

        # Display the duration of game on the window
        duration_total = self.steps // self.metadata["render_fps"]
        duration_min = duration_total // 60
        duration_sec = duration_total - duration_min * 60
        time_surf = self.font.render(f"Time: {duration_min:0>2d}:{duration_sec:0>2d}", True, (0, 0, 0))
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
