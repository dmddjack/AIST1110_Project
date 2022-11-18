import numpy as np
import pygame
import gym
from gym import spaces

from .assets import Enemy, Heart, Player


class TankWar(gym.Env):
    metadata = {"render_modes": ("human", "rgb_array"), "render_fps": 30}

    def __init__(self, render_mode: str | None = None, starting_hp: int = 3) -> None:
        self.starting_hp = starting_hp  # The starting health point (HP) of the player
        self.window_width, self.window_height = 600, 400  # The size of the PyGame window

        # Observation is a 51-element list: the player's location, a max. of 4 enemies' locations, a max. of 20 enemy's bullets' locations, player's gun's remaining reloading time
        self.observation_space = spaces.Box(0, max(self.window_width, self.window_height), shape=(51,), dtype=np.float32)

        # We have 10 actions: up, down, left, right, shoot, up and shoot, down and shoot, left and shoot, right and shoot, do nothing
        self.action_space = spaces.Discrete(10)

        assert render_mode is None or render_mode in self.metadata["render_modes"], f"Invalid render mode, has to be {', '.join(self.metadata['render_modes'])}"
        self.render_mode = render_mode

        # The following will remain None iff "rgb_array" mode is used
        self.window = None
        self.clock = None
        self.font = None

    def _get_obs(self) -> np.ndarray:
        pass

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        # Seed self.np_random
        super().reset(seed=seed)

        # Reset all counting variables
        self.steps = 0
        self.score = 0
        self.hp = self.starting_hp

        # Create a sprite group for all sprites
        self.all_sprites = pygame.sprite.Group()

        # Create the player
        self._create_player()

        # Create a sprite group for enemies
        self.enemies = pygame.sprite.Group()

        # Create enemies
        self._create_enemy()

        # Create sprite groups for the player's and enemies' bullets
        self.player_bullets = pygame.sprite.Group()
        self.enemy_bullets = pygame.sprite.Group()

        # Create a sprite group for hearts
        self.hearts = pygame.sprite.Group()

        # Create hearts and add them to self.hearts and self.all_sprites
        for i in range(1, self.starting_hp + 1):
            heart = Heart(self.window_width, i)
            self.hearts.add(heart)
            self.all_sprites.add(heart)

        obs = self._get_obs()
        
        # Create a placeholder for additional information
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def _create_player(self) -> None:
        """An internal function that creates one player at a random position in the middle of the window"""

        player_start_x = int(self.np_random.integers(self.window_width * 0.3, self.window_width * 0.7, size=1))
        player_start_y = int(self.np_random.integers(self.window_height * 0.3, self.window_height * 0.7, size=1))
        player_start_angle = self.np_random.choice((0, 90, 180, 270))
        self.player = Player(player_start_x, player_start_y, player_start_angle, self.window_width, self.window_height, self._fps_to_speed(4))
        self.all_sprites.add(self.player)

    def _score_to_enemy(self) -> tuple[int, int, float]:
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
        return enemy_n, self._fps_to_speed(enemy_speed), enemy_shoot_intvl

    def _create_enemy(self) -> tuple[int, float]:
        """An internal function that creates sufficient enemies at random positions on the borders"""

        enemy_n, enemy_speed, enemy_shooting_interval = self._score_to_enemy()
        for _ in range(enemy_n - len(self.enemies)):
            # Keep recreating an enemy until it does not collide with the player or other enemies
            overlapped = True
            while overlapped:
                new_enemy_start_angle = self.np_random.choice((0, 90, 180, 270))
                if new_enemy_start_angle == 0:
                    enemy_start_x = int(self.np_random.integers(0, self.window_width, size=1))
                    enemy_start_y = self.window_height
                elif new_enemy_start_angle == 90:
                    enemy_start_x = self.window_width
                    enemy_start_y = int(self.np_random.integers(0, self.window_height, size=1))
                elif new_enemy_start_angle == 180:
                    enemy_start_x = int(self.np_random.integers(0, self.window_width, size=1))
                    enemy_start_y = 0
                else:
                    enemy_start_x = 0
                    enemy_start_y = int(self.np_random.integers(0, self.window_height, size=1))
                enemy = Enemy(enemy_start_x, enemy_start_y, new_enemy_start_angle, self.window_width, self.window_height, enemy_speed, self.steps)
                
                # Check if the new enemy collides with the player or other enemies
                if not (pygame.sprite.collide_rect(enemy, self.player) or pygame.sprite.spritecollideany(enemy, self.enemies)):
                    overlapped = False

            # Add the new enemy to self.enemies and self.all_sprites
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)

        return enemy_speed, enemy_shooting_interval

    def _fps_to_speed(self, original_speed: int) -> int:
        """An internal function that converts the original speed, which was based on a framerate of 30, to a speed that fits any framerate"""

        return 30 * original_speed // self.metadata["render_fps"]

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
            elif enemy_angle == 90: # Move left
                enemy_dx = -1
            elif enemy_angle == 180: # Move down
                enemy_dy = 1
            else: # Move right
                enemy_dx = 1
            
            # Update the enemy's position
            enemy_is_outside, enemy_possible_new_angles = enemy.update(enemy_dx, enemy_dy, enemy_angle)

            # Ensure the enemy does not stuck at the border by reversing their direction
            if enemy_is_outside:
                enemy_new_angle = self.np_random.choice(enemy_possible_new_angles)
                enemy.update(0, 0, enemy_new_angle)
                enemy.last_rotate = self.steps

            # Shoot a bullet from the enemy's position
            self._enemy_shoot(enemy, enemy_angle, enemy_shoot_intvl)

        ##### Better algorithm is required
        # Reverse the directions of two enemies when they collide with each other
        enemy_collision = pygame.sprite.groupcollide(self.enemies, self.enemies, dokilla=False, dokillb=False)
        for enemy in [enemy for enemies in enemy_collision.values() if len(enemies) > 1 for enemy in enemies]:
            # enemy_old_angle = enemy.angle
            # if enemy_old_angle == 0:
            #     enemy.update(0, 1, 180)
            # elif enemy_old_angle == 90:
            #     enemy.update(1, 0, 270)
            # elif enemy_old_angle == 180:
            #     enemy.update(0, -1, 0)
            # else:
            #     enemy.update(-1, 0, 90)
            # enemy.last_rotate = self.steps
            enemy.kill()

        # Move the player's and enemies' bullets
        for bullets in (self.player_bullets, self.enemy_bullets):
            for bullet in bullets:
                bullet.move()

                # Remove the bullet if it is outside the window
                if bullet.rect.right < 0 or bullet.rect.left > self.window_width or bullet.rect.bottom <= 0 or bullet.rect.top >= self.window_height:
                    bullet.kill()

        # Terminate the episode if the player has collided into any of the enemies or any of the enemies' bullets
        if pygame.sprite.spritecollideany(self.player, self.enemies) or len(pygame.sprite.spritecollide(self.player, self.enemy_bullets, dokill=True)):
            reward -= 50
            self.hp -= 1

            # Remove the player
            self.player.kill()

            if self.hp == 0:
                terminated = True
            else:
                # Remove all enemies to ensure the player will not be killed at spawn
                for enemy in self.enemies:
                    enemy.kill()

                # Remove all bullets
                for bullets in (self.player_bullets, self.enemy_bullets):
                    for bullet in bullets:
                        bullet.kill()

                # Respawn the player at a randomly generated location
                player_respawn_x = int(self.np_random.integers(self.window_width * 0.3, self.window_width * 0.7, size=1))
                player_respawn_y = int(self.np_random.integers(self.window_height * 0.3, self.window_height * 0.7, size=1))
                player_respawn_angle = self.np_random.choice((0, 90, 180, 270))
                self.player.rect = self.player.surf.get_rect(center=(player_respawn_x, player_respawn_y))
                self.player.update(0, 0, player_respawn_angle)
                self.player.last_shoot = 0

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

        obs = self._get_obs()
        
        # Create a placeholder for additional information
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, info

    def _player_shoot(self, angle: int) -> None:
        # The player can shoot with a predefined interval
        if self.player.last_shoot == 0 or self.steps - self.player.last_shoot >= self.metadata["render_fps"]:
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

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Tank War")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        if self.font is None and self.render_mode == "human":
            self.font = pygame.font.SysFont('Garamond', 25)
            
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        
        if self.render_mode == "human":
            # Set the background
            bg = pygame.image.load("./images/background.png")
            bg.set_alpha(200)
            canvas.blit(bg, (0, 0))

        # Draw all sprites
        for sprite in self.all_sprites:
            canvas.blit(sprite.surf, sprite.rect)

        if self.render_mode == "human":
            # Display the score on the window
            score_surf = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
            canvas.blit(score_surf, (5, 5))

            # Display the duration of game on the window
            duration_total = self.steps // self.metadata["render_fps"]
            duration_min = duration_total // 60
            duration_sec = duration_total - duration_min * 60
            time_surf = self.font.render(f"Time: {duration_min:0>2d}:{duration_sec:0>2d}", True, (0, 0, 0))
            canvas.blit(time_surf, (5, 25))

        # Display the player's gun's remaining reloading time as a shrinking rectangle
        if self.player.last_shoot != 0:
            reload_bar_len = 80 * (self.metadata["render_fps"] - (self.steps - self.player.last_shoot)) // self.metadata["render_fps"]
            pygame.draw.rect(canvas, (230, 230, 230), (self.window_width - reload_bar_len - 15, self.window_height - 20, reload_bar_len, 10))

        if self.render_mode == "human":
            # Draw the canvas to the PyGame window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Ensure the rendering occurs at the predefined framerate
            self.clock.tick(self.metadata["render_fps"])
        else: # Return an RGB array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
            
    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
