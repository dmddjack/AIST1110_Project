#!/usr/bin/env python3

import gym
import numpy as np
import pygame
from gym import spaces

from .assets import Audios, Background, Enemy, Heart, Player


class TankWar(gym.Env):
    metadata = {"render_modes": ("human", "rgb_array"), "render_fps": 30}

    def __init__(self, render_mode: str | None, 
            starting_hp: int, difficulty: int) -> None:
        # The starting health point (HP) of the player
        self.starting_hp = starting_hp

        # The difficulty of AI
        self.difficulty = difficulty

        # The size of the pygame window
        self.window_width, self.window_height = 450, 350

        # All possible angles of the tanks and bullets
        self.angles = (0, 90, 180, 270)

        # The moving speed of the player based on a framerate of 30
        self.player_speed = 4

        # The minimum interval between the player's last shot and next shot
        self.player_shoot_intvl = 1

        # The maximum number of player's bullets
        self.max_player_bullets = 6

        # The maximum number of enemies
        self.max_enemies = 4

        # The maximum number of enemies' bullets
        self.max_enemy_bullets = self.max_enemies * 3

        # The reward when the player kills an enemy
        self.enemy_killed_reward = 5

        # The reward when the player is killed by the enemies
        self.player_killed_reward = -1

        # Normalized observation: the player's location, 
        # the player's bullets' location, 
        # enemies' locations, enemy's bullets' locations, 
        # player's cannon's remaining reloading time
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=((1 + self.max_player_bullets + self.max_enemies + \
                self.max_enemy_bullets) * 3 + 1,), 
            dtype=np.float32,
        )

        # print(self.observation_space.sample())  # For testing purposes

        # We have 10 actions: up, down, left, right, shoot, up and shoot, 
        # down and shoot, left and shoot, right and shoot, do nothing
        self.action_space = spaces.Discrete(10)

        assert (
            render_mode is None or 
            render_mode in self.metadata["render_modes"]
        )

        self.render_mode = render_mode

        self.pygame_initialized = False
        
        self.font = None
        self.background = None

        # The following will remain None iff "rgb_array" mode is used
        self.window = None
        self.clock = None

    def _get_observation(self) -> np.ndarray:
        # Get the player's observation
        player_observation = self.player.get_observation()

        # Get all player's bullets' observation
        if len(self.player_bullets) > 0:
            player_bullets_observation = np.concatenate(
                [bullet.get_observation() for bullet in self.player_bullets]
            )
            player_bullets_observation = np.pad(
                player_bullets_observation,
                (0, self.max_player_bullets * 3 \
                    - len(player_bullets_observation)), 
                "constant",
                constant_values=(0,),
            )
        else:
            player_bullets_observation = np.full(
                (self.max_player_bullets * 3,), 
                0, 
                dtype=np.float32,
            )

        # Get all enemies' observation
        if len(self.enemies) > 0:
            enemies_observation = np.concatenate(
                [enemy.get_observation() for enemy in self.enemies]
            )
            enemies_observation = np.pad(
                enemies_observation,
                (0, self.max_enemies * 3 - len(enemies_observation)), 
                "constant",
                constant_values=(0,),
            )
        else:
            enemies_observation = np.full(
                (self.max_enemies * 3,), 
                0, 
                dtype=np.float32,
            )

        # Get all enemies' bullets' observation
        if len(self.enemy_bullets) > 0:
            enemy_bullets_observation = np.concatenate(
                [bullet.get_observation() for bullet in self.enemy_bullets]
            )
            enemy_bullets_observation = np.pad(
                enemy_bullets_observation,
                (0, self.max_enemy_bullets * 3 \
                    - len(enemy_bullets_observation)),
                "constant",
                constant_values=(0,),
            )
        else:
            enemy_bullets_observation = np.full(
                (self.max_enemy_bullets * 3,), 
                0, 
                dtype=np.float32,
            )

        # Get the player's cannon's remaining reloading time
        player_cannon_reload_time = (
            0 if self.player.last_shoot == 0
            else max(
                0,
                1 - (self.steps - self.player.last_shoot) \
                    / (self.metadata["render_fps"] * self.player_shoot_intvl),
            )
        )
        player_cannon_reload_time = np.array(
            [player_cannon_reload_time],
            dtype=np.float32,
        )

        # Concatenate all NumPy arrays
        observation = np.concatenate(
            [
                player_observation, player_bullets_observation, 
                enemies_observation, enemy_bullets_observation, 
                player_cannon_reload_time,
            ],
            dtype=np.float32,
        )

        # print(observation)  # For testing purposes

        return observation

    def reset(self, seed: int | None = None, 
            options=None) -> tuple[np.ndarray, dict]:
        # Seed self.np_random
        super().reset(seed=seed)

        # Reset all counting variables
        self.steps = 0
        self.score = 0
        self.hp = self.starting_hp

        # Create a sprite group for all sprites except hearts
        self.all_sprites = pygame.sprite.Group()

        # Create the player
        self._create_player()

        # Create a sprite group for enemies
        self.enemies = pygame.sprite.Group()

        # Create sufficient enemies
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

        observation = self._get_observation()

        # Create a placeholder for additional information
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _create_player(self) -> None:
        """
        An internal function that creates one player at a random location 
        in the middle of the window.
        """

        # Randomly generate a starting location in the middle of the window
        player_start_x = int(
            self.np_random.integers(
                self.window_width * 0.3, self.window_width * 0.7, size=1
            )
        )
        player_start_y = int(
            self.np_random.integers(
                self.window_height * 0.3, self.window_height * 0.7, size=1
            )
        )

        # Randomly generate a starting angle
        player_start_angle = self.np_random.choice(self.angles)

        # Create a new player
        self.player = Player(
            window_width=self.window_width,
            window_height=self.window_height,
            start_x=player_start_x,
            start_y=player_start_y,
            start_angle=player_start_angle,
            speed=self._fps_to_speed(self.player_speed, self.metadata["render_fps"]),
        )

        # Add the new player to self.all_sprites
        self.all_sprites.add(self.player)

    @staticmethod
    def _score_to_enemy(score: int, max_enemies: int, render_fps: int) -> tuple[int, int, float]:
        """
        An internal function that maps the current score to the behaviour 
        of the enemies.
        """

        if score < 5:
            enemy_n, enemy_speed, enemy_shoot_intvl = 1, 2, 2
        elif score < 10:
            enemy_n, enemy_speed, enemy_shoot_intvl = 2, 2, 2
        elif score < 15:
            enemy_n, enemy_speed, enemy_shoot_intvl = 3, 2, 1.5
        elif score < 20:
            enemy_n, enemy_speed, enemy_shoot_intvl = 3, 3, 1.5
        elif score < 25:
            enemy_n, enemy_speed, enemy_shoot_intvl = 4, 3, 1.5
        else:
            enemy_n, enemy_speed, enemy_shoot_intvl = 4, 3, 1.2

        return min(enemy_n, max_enemies), \
               TankWar._fps_to_speed(enemy_speed, render_fps), \
               enemy_shoot_intvl

    def _create_enemy(self) -> tuple[int, float]:
        """
        An internal function that creates sufficient enemies at 
        random locations on the borders.
        """

        enemy_n, enemy_speed, enemy_shoot_intvl = \
            self._score_to_enemy(self.score, 
                                 self.max_enemies, 
                                 self.metadata["render_fps"])
        for _ in range(enemy_n - len(self.enemies)):
            # Keep recreating an enemy until it does not collide 
            # with the player or other enemies
            overlapped = True
            while overlapped:
                # Randomly generate a starting angle
                new_enemy_start_angle = self.np_random.choice(self.angles)

                # Choose which border to start from based on the starting angle
                if new_enemy_start_angle == 0:
                    enemy_start_x = int(
                        self.np_random.integers(0, self.window_width, size=1)
                    )
                    enemy_start_y = self.window_height
                elif new_enemy_start_angle == 90:
                    enemy_start_x = self.window_width
                    enemy_start_y = int(
                        self.np_random.integers(0, self.window_height, size=1)
                    )
                elif new_enemy_start_angle == 180:
                    enemy_start_x = int(
                        self.np_random.integers(0, self.window_width, size=1)
                    )
                    enemy_start_y = 0
                else:
                    enemy_start_x = 0
                    enemy_start_y = int(
                        self.np_random.integers(0, self.window_height, size=1)
                    )

                # Create a new enemy
                enemy = Enemy(
                    window_width=self.window_width,
                    window_height=self.window_height,
                    start_x=enemy_start_x,
                    start_y=enemy_start_y,
                    start_angle=new_enemy_start_angle,
                    speed=enemy_speed,
                    creation_step=self.steps,
                )

                # Check if the new enemy collides with the player or 
                # other enemies
                if not (pygame.sprite.collide_rect(enemy, self.player) or 
                        pygame.sprite.spritecollideany(enemy, self.enemies)):
                    overlapped = False

            # Add the new enemy to self.enemies and self.all_sprites
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)

        return enemy_speed, enemy_shoot_intvl

    @staticmethod
    def _fps_to_speed(original_speed: int, render_fps: int) -> int:
        """
        An internal function that converts the original speed, 
        which was based on a framerate of 30, to a speed that fits 
        any framerate.
        """

        return 30 * original_speed // render_fps

    def step(self, action: int | None):
        self.steps += 1
        reward = 0
        terminated = False

        if self.pygame_initialized:
            # Set the player's engine sound volume to normal
            self.tank_engine_sound.set_volume(0.4)

        if action is not None:
            """Step 1: Move the player according to the action"""

            player_dx, player_dy, player_new_angle = 0, 0, self.player.angle
            player_shoot = False
            if action == 0 or action == 5:  # Move up
                player_dy = -1
                player_new_angle = 0
                if action == 5:  # Shoot while moving up
                    player_shoot = True
            elif action == 1 or action == 6:  # Move down
                player_dy = 1
                player_new_angle = 180
                if action == 6:  # Shoot while moving down
                    player_shoot = True
            elif action == 2 or action == 7:  # Move left
                player_dx = -1
                player_new_angle = 90
                if action == 7:  # Shoot while moving left
                    player_shoot = True
            elif action == 3 or action == 8:  # Move right
                player_dx = 1
                player_new_angle = 270
                if action == 8:  # Shoot while moving right
                    player_shoot = True
            elif action == 9:  # Shoot while not moving
                player_shoot = True

            if action != 4 and action != 9:
                # Update the player's location
                self.player.update(
                    dx=player_dx,
                    dy=player_dy,
                    new_angle=player_new_angle
                )

                if self.pygame_initialized:
                    # Make the player's engine sound louder
                    self.tank_engine_sound.set_volume(0.9)

            """
            Step 2: Shoot a bullet from the player's location if player_shoot 
            is true
            """

            if player_shoot:
                self._player_shoot(angle=self.player.angle)

        """Step 3: Create sufficient enemies based on self._score_to_enemy()"""

        enemy_speed, enemy_shoot_intvl = self._create_enemy()

        """Step 4: Move the enemies and let them shoot"""

        for enemy in self.enemies:
            enemy.speed = enemy_speed
            enemy_dx, enemy_dy, enemy_new_angle = 0, 0, enemy.angle

            # Rotates an enemy with an interval of not less than 2 seconds 
            # and a probability of 2% (based on a framerate of 30)
            if (self.steps - enemy.last_rotate >= \
                    self.metadata["render_fps"] * 2 and 
                    self.np_random.random() < self._fps_to_prob(0.02)):
                enemy_new_angle = self.np_random.choice(
                    [angle for angle in self.angles if angle != enemy.angle]
                )
                enemy.last_rotate = self.steps

            if enemy_new_angle == 0:  # Move up
                enemy_dy = -1
            elif enemy_new_angle == 90:  # Move left
                enemy_dx = -1
            elif enemy_new_angle == 180:  # Move down
                enemy_dy = 1
            else:  # Move right
                enemy_dx = 1

            # Update the enemy's location
            enemy_touches_border, enemy_correction_angles = enemy.update(
                dx=enemy_dx,
                dy=enemy_dy,
                new_angle=enemy_new_angle
            )

            # Ensure the enemy does not stuck at the border by 
            # reversing its direction
            if enemy_touches_border:
                enemy_new_angle = self.np_random.choice(enemy_correction_angles)
                enemy.update(0, 0, enemy_new_angle)
                enemy.last_rotate = self.steps

            # Shoot a bullet from the enemy's location
            self._enemy_shoot(enemy, enemy.angle, enemy_shoot_intvl)

        """
        Step: 5 Handle situations where two enemies collide with 
        each other
        """

        enemy_collision = pygame.sprite.groupcollide(
            self.enemies, self.enemies, dokilla=False, dokillb=False
        )
        for enemy in [enemy
                for enemies in enemy_collision.values()
                if len(enemies) > 1
                for enemy in enemies]:
            # Reverse the directions of two enemies when they collide 
            # with each other
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

            # Kill the enemy
            enemy.kill()

        """Step 6: Move the player's and enemies' bullets"""

        for bullets in (self.player_bullets, self.enemy_bullets):
            for bullet in bullets:
                bullet.move()

                # Remove the bullet if it is outside the window
                if (bullet.rect.right < 0 or 
                        bullet.rect.left > self.window_width or 
                        bullet.rect.bottom <= 0 or 
                        bullet.rect.top >= self.window_height):
                    bullet.kill()

        """Step 7: Remove the player's bullet if it hits an enemy"""

        for bullet in self.player_bullets:
            if pygame.sprite.spritecollide(
                    bullet, 
                    self.enemies, 
                    dokill=True):
                reward += self.enemy_killed_reward
                self.score += 1
                bullet.kill()

                if self.pygame_initialized:
                    # Play the explosion sound effect
                    self.explosion_sound.play()

        """Step 8: Remove the player's bullet if it hits an enemy's bullet"""

        for bullet in self.player_bullets:
            if pygame.sprite.spritecollide(
                    bullet, 
                    self.enemy_bullets, 
                    dokill=True):
                bullet.kill()

        """
        Step 9: Deduct 1 HP if the player has collided with 
        any of the enemies or any of the enemies' bullets, terminate the 
        episode if self.hp == 0
        """

        if (pygame.sprite.spritecollide(
                self.player, 
                self.enemies, 
                dokill=True,
                ) or 
                pygame.sprite.spritecollide(
                    self.player, 
                    self.enemy_bullets, 
                    dokill=True
                )):
            reward += self.player_killed_reward
            self.hp -= 1

            # Kill the player
            self.player.kill()

            if self.pygame_initialized:
                # Play the explosion sound effect
                self.explosion_sound.play()

            if self.hp == 0:
                terminated = True
            else:
                # Kill all enemies to ensure the player will not be killed 
                # at spawn
                for enemy in self.enemies:
                    enemy.kill()

                # Remove all bullets to make the window look nice and 
                # ensure the player will not be killed at spawn
                for bullets in (self.player_bullets, self.enemy_bullets):
                    for bullet in bullets:
                        bullet.kill()

                # Respawn the player
                self._create_player()

            # Remove the leftmost heart
            self.hearts.sprites()[-1].kill()

        """Step 10: Terminate the episode if self.score >= 26"""
        # if self.score >= 26:
        #     terminated = True
        
        observation = self._get_observation()

        # Create a placeholder for additional information
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _fps_to_prob(self, original_prob: float) -> float:
        """
        An internal function that converts the original probability, 
        which was based on a framerate of 30, to a probability that 
        fits any framerate.
        """

        return 30 * original_prob / self.metadata["render_fps"]

    def _player_shoot(self, angle: int) -> None:
        """
        An internal function that controls how the player shoots. 
        The player can shoot at the beginning or for every one second.
        """

        if (len(self.player_bullets) < self.max_player_bullets and 
                (self.player.last_shoot == 0 or 
                    self.steps - self.player.last_shoot \
                        >= self.metadata["render_fps"] \
                            * self.player_shoot_intvl)):
            self.player.last_shoot = self.steps

            # Create a new bullet for the player
            player_bullet = self.player.bullet(
                window_width=self.window_width,
                window_height=self.window_height,
                tank_size=self.player.surf.get_size(),
                tank_center=self.player.rect.center,
                angle=angle,
                speed=self.player.speed + self._fps_to_speed(3, self.metadata["render_fps"]),
            )

            # Add the player's new bullet to self.player_bullets and 
            # self.all_sprites
            self.player_bullets.add(player_bullet)
            self.all_sprites.add(player_bullet)

            if self.pygame_initialized:
                # Play the cannon firing sound effect
                self.cannon_fire_sound.play()

    def _enemy_shoot(self, enemy: Enemy, angle: int, interval: int) -> None:
        """
        An internal function that controls how an enemy shoots. 
        It shoots with a predefined interval and 
        a probability of 5% (based on a framerate on 30).
        """

        if (len(self.enemy_bullets) < self.max_enemy_bullets and 
                self.steps - enemy.last_shoot >= \
                    self.metadata["render_fps"] * interval and 
                self.np_random.random() < self._fps_to_prob(0.05)):

            enemy.last_shoot = self.steps

            # Create a new bullet for the enemy
            enemy_bullet = enemy.bullet(
                window_width=self.window_width,
                window_height=self.window_height,
                tank_size=enemy.surf.get_size(),
                tank_center=enemy.rect.center,
                angle=angle,
                speed=enemy.speed + self._fps_to_speed(2, self.metadata["render_fps"]),
            )

            # Add the enemy's new bullet to self.enemy_bullets and 
            # self.all_sprites
            self.enemy_bullets.add(enemy_bullet)
            self.all_sprites.add(enemy_bullet)

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        if not self.pygame_initialized:
            # Initialize pygame
            pygame.init()

            # Initialize sound module
            pygame.mixer.init()

            # Load and play the background music
            pygame.mixer.music.load(Audios.background_music)
            pygame.mixer.music.play(loops=-1)

            # Load and keep looping the tank engine sound effect
            self.tank_engine_sound = pygame.mixer.Sound(Audios.tank_engine_sound)
            self.tank_engine_sound.set_volume(0.4)
            self.tank_engine_sound.play(loops=-1)

            # Load the cannon firing sound effect
            self.cannon_fire_sound = pygame.mixer.Sound(Audios.cannon_fire_sound)

            # Load the explosion sound effect
            self.explosion_sound = pygame.mixer.Sound(Audios.explosion_sound)
            self.explosion_sound.set_volume(0.8)

            self.pygame_initialized = True

        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Tank War")
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None:
            # Initialize the font
            self.font = pygame.font.SysFont("Garamond", 25)

        if self.background is None:
            # Load the background
            self.background = Background()

        # Create a surface to hold all elements
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # Set the background
        canvas.blit(self.background.surf, (0, 0))

        # Draw all sprites
        for sprite in self.all_sprites:
            canvas.blit(sprite.surf, sprite.rect)

        # Draw all hearts
        for heart in self.hearts:
            canvas.blit(heart.surf, heart.rect)

        # Display the score on the window
        score_surf = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        canvas.blit(score_surf, (5, 5))

        # Display the duration of game on the window
        duration_total = self.steps // self.metadata["render_fps"]
        duration_min = duration_total // 60
        duration_sec = duration_total - duration_min * 60
        time_surf = self.font.render(
            f"Time: {duration_min:0>2d}:{duration_sec:0>2d}", True, (0, 0, 0)
        )
        canvas.blit(time_surf, (5, 25))

        # Display the player's cannon's remaining reloading time as a 
        # shrinking rectangle
        if self.player.last_shoot != 0:
            reload_bar_len = max(
                0,
                80 * (self.metadata["render_fps"] * self.player_shoot_intvl \
                    - (self.steps - self.player.last_shoot)) \
                    // (self.metadata["render_fps"] * self.player_shoot_intvl),
            )
            pygame.draw.rect(
                canvas,
                (230, 230, 230),
                (
                    self.window_width - reload_bar_len - 15,
                    self.window_height - 20,
                    reload_bar_len,
                    10,
                ),
            )

        if self.render_mode == "human":
            # Draw the canvas to the pygame window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # Ensure the rendering occurs at the predefined framerate
            self.clock.tick(self.metadata["render_fps"])
        else:  # Return an RGB array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()

        if self.pygame_initialized:
            # Stop and quit the sound module
            pygame.mixer.music.stop()
            pygame.mixer.quit()

            # Quit pygame
            pygame.quit()
