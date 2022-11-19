import gym
import numpy as np
import pygame
from gym import spaces

from .assets import Enemy, Heart, Player


class TankWar(gym.Env):
    metadata = {"render_modes": ("human", "rgb_array"), "render_fps": 30}

    def __init__(self, render_mode: str | None = None, starting_hp: int = 3) -> None:
        # The starting health point (HP) of the player
        self.starting_hp = starting_hp

        # The size of the PyGame window
        self.window_width, self.window_height = 600, 400

        # All possible angles of the tanks and bullets
        self.angles = (0, 90, 180, 270)

        # The moving speed of the player based on a framerate of 30
        self.player_speed = 4

        # The maximum number of enemies
        self.max_enemies = 4  

        # The maximum number of enemies' bullets
        self.max_enemy_bullets = 20

        # The reward when the player kills an enemy
        self.enemy_killed_reward = 1

        # The reward when the player is killed by the enemies
        self.player_killed_reward = -5

        """
        Observation is a 51-element normalized list: the player's location, 
        a max. of 4 enemies' locations, a max. of 20 enemy's bullets' locations, 
        player's gun's remaining reloading time
        """
        self.observation_space = spaces.Box(-1, 1, shape=(51,), dtype=np.float32)

        """
        We have 10 actions: up, down, left, right, shoot, up and shoot, 
        down and shoot, left and shoot, right and shoot, do nothing
        """
        self.action_space = spaces.Discrete(10)

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        ), f"Invalid render mode, has to be {', '.join(self.metadata['render_modes'])}"
        self.render_mode = render_mode

        # The following will remain None iff "rgb_array" mode is used
        self.window = None
        self.clock = None
        self.font = None

    def _get_obs(self) -> np.ndarray:
        player_loc = self.player.get_location()
        if len(self.enemies) > 0:
            enemies_loc = np.concatenate(
                [enemy.get_location() for enemy in self.enemies]
            )
            enemies_loc = np.pad(
                enemies_loc,
                (0, self.max_enemies * 2 - len(enemies_loc)),
                "constant",
                constant_values=(-1,),
            )
        else:
            enemies_loc = np.full((self.max_enemies * 2,), -1)
        if len(self.enemy_bullets) > 0:
            enemy_bullets_loc = np.concatenate(
                [bullet.get_location() for bullet in self.enemy_bullets]
            )
            enemy_bullets_loc = np.pad(
                enemy_bullets_loc,
                (0, self.max_enemy_bullets * 2 - len(enemy_bullets_loc)),
                "constant",
                constant_values=(-1,),
            )
        else:
            enemy_bullets_loc = np.full((self.max_enemy_bullets * 2,), -1)
        player_gun_reload_time = (
            0 if self.player.last_shoot == 0
            else max(
                0,
                1 - (self.steps - self.player.last_shoot) / (self.metadata["render_fps"] * 1),
            )
        )
        player_gun_reload_time = np.array([player_gun_reload_time])
        obs = np.concatenate(
            [player_loc, enemies_loc, enemy_bullets_loc, player_gun_reload_time],
            dtype=np.float32,
        )
        # print(obs)  # For testing purposes
        return obs

    def reset(self, seed: int | None = None, options=None) -> tuple[np.ndarray, dict]:
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

        obs = self._get_obs()

        # Create a placeholder for additional information
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def _create_player(self) -> None:
        """
        An internal function that creates one player at a random location 
        in the middle of the window.
        """

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
        player_start_angle = self.np_random.choice((0, 90, 180, 270))
        self.player = Player(
            window_width=self.window_width,
            window_height=self.window_height,
            start_x=player_start_x,
            start_y=player_start_y,
            start_angle=player_start_angle,
            speed=self._fps_to_speed(self.player_speed),
        )
        self.all_sprites.add(self.player)

    def _score_to_enemy(self) -> tuple[int, int, float]:
        """
        An internal function that maps the current score to the behaviour 
        of the enemies.
        """

        enemy_n, enemy_speed, enemy_shoot_intvl = 0, 0, 0
        if self.score < 5 * self.enemy_killed_reward:
            enemy_n, enemy_speed, enemy_shoot_intvl = 1, 2, 2
        elif self.score < 10 * self.enemy_killed_reward:
            enemy_n, enemy_speed, enemy_shoot_intvl = 1, 3, 2
        elif self.score < 15 * self.enemy_killed_reward:
            enemy_n, enemy_speed, enemy_shoot_intvl = 2, 2, 2
        elif self.score < 20 * self.enemy_killed_reward:
            enemy_n, enemy_speed, enemy_shoot_intvl = 2, 3, 2
        elif self.score < 25 * self.enemy_killed_reward:
            enemy_n, enemy_speed, enemy_shoot_intvl = 3, 2, 1.5
        elif self.score < 30 * self.enemy_killed_reward:
            enemy_n, enemy_speed, enemy_shoot_intvl = 3, 3, 1.5
        elif self.score < 35 * self.enemy_killed_reward:
            enemy_n, enemy_speed, enemy_shoot_intvl = 4, 2, 1.5
        else:
            enemy_n, enemy_speed, enemy_shoot_intvl = 4, 3, 1.5
        return enemy_n, self._fps_to_speed(enemy_speed), enemy_shoot_intvl

    def _create_enemy(self) -> tuple[int, float]:
        """
        An internal function that creates sufficient enemies at 
        random locations on the borders.
        """

        enemy_n, enemy_speed, enemy_shoot_intvl = self._score_to_enemy()
        for _ in range(enemy_n - len(self.enemies)):
            # Keep recreating an enemy until it does not collide with the player or other enemies
            overlapped = True
            while overlapped:
                new_enemy_start_angle = self.np_random.choice((0, 90, 180, 270))
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
                enemy = Enemy(
                    window_width=self.window_width,
                    window_height=self.window_height,
                    start_x=enemy_start_x,
                    start_y=enemy_start_y,
                    start_angle=new_enemy_start_angle,
                    speed=enemy_speed,
                    creation_step=self.steps,
                )

                # Check if the new enemy collides with the player or other enemies
                if not (
                    pygame.sprite.collide_rect(enemy, self.player)
                    or pygame.sprite.spritecollideany(enemy, self.enemies)
                ):
                    overlapped = False

            # Add the new enemy to self.enemies and self.all_sprites
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)

        return enemy_speed, enemy_shoot_intvl

    def _fps_to_speed(self, original_speed: int) -> int:
        """
        An internal function that converts the original speed, 
        which was based on a framerate of 30, to a speed that fits 
        any framerate.
        """

        return 30 * original_speed // self.metadata["render_fps"]

    def step(self, action):
        self.steps += 1
        reward = 0
        terminated = False

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
            elif action == 4:  # Shoot while not moving
                player_shoot = True

            if action != 9:
                # Update the player's location
                self.player.update(
                    dx=player_dx,
                    dy=player_dy,
                    new_angle=player_new_angle
                )

            """Step 2: Shoot a bullet from the player's location if player_shoot is true"""

            if player_shoot:
                self._player_shoot(angle=self.player.angle)

        """Step 3: Create sufficient enemies based on self._score_to_enemy()"""

        enemy_speed, enemy_shoot_intvl = self._create_enemy()

        """Step 4: Move the enemies and let them shoot"""

        for enemy in self.enemies:
            enemy.speed = enemy_speed
            enemy_dx, enemy_dy, enemy_new_angle = 0, 0, enemy.angle

            # Rotates an enemy with an interval of not less than 3 seconds and a probability of 2% (based on a framerate of 30)
            if (
                self.steps - enemy.last_rotate >= self.metadata["render_fps"] * 3
                and self.np_random.random() < self._fps_to_prob(0.02)
            ):
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

            # Ensure the enemy does not stuck at the border by reversing its direction
            if enemy_touches_border:
                enemy_new_angle = self.np_random.choice(enemy_correction_angles)
                enemy.update(0, 0, enemy_new_angle)
                enemy.last_rotate = self.steps

            # Shoot a bullet from the enemy's location
            self._enemy_shoot(enemy, enemy.angle, enemy_shoot_intvl)

        """Step: 5 Handle situations where two enemies collide with each other"""

        enemy_collision = pygame.sprite.groupcollide(
            self.enemies, self.enemies, dokilla=False, dokillb=False
        )
        for enemy in [
            enemy
            for enemies in enemy_collision.values()
            if len(enemies) > 1
            for enemy in enemies
        ]:
            # Reverse the directions of two enemies when they collide with each other
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

        """Step 6: Move the player's and enemies' bullets"""

        for bullets in (self.player_bullets, self.enemy_bullets):
            for bullet in bullets:
                bullet.move()

                # Remove the bullet if it is outside the window
                if (
                    bullet.rect.right < 0
                    or bullet.rect.left > self.window_width
                    or bullet.rect.bottom <= 0
                    or bullet.rect.top >= self.window_height
                ):
                    bullet.kill()

        """Step 7: Remove the player's bullet if it hits an enemy"""

        for bullet in self.player_bullets:
            if pygame.sprite.spritecollide(bullet, self.enemies, dokill=True):
                bullet.kill()
                reward += self.enemy_killed_reward
                self.score += self.enemy_killed_reward

        """Step 8: Remove the player's bullet if it hits an enemy's bullet"""

        for bullet in self.player_bullets:
            if pygame.sprite.spritecollide(bullet, self.enemy_bullets, dokill=True):
                bullet.kill()

        """
        Step 9: Terminate the episode if the player has collided with 
        any of the enemies or any of the enemies' bullets
        """

        if (
            pygame.sprite.spritecollide(self.player, self.enemies, dokill=True)
            or pygame.sprite.spritecollide(self.player, self.enemy_bullets, dokill=True)
        ):
            reward += self.player_killed_reward
            self.hp -= 1

            # Remove the player
            self.player.kill()

            if self.hp == 0:
                terminated = True
            else:
                # Remove all enemies to ensure the player will not be killed at spawn
                for enemy in self.enemies:
                    enemy.kill()

                # Remove all bullets to make the window look nice and ensure the player will not be killed at spawn
                for bullets in (self.player_bullets, self.enemy_bullets):
                    for bullet in bullets:
                        bullet.kill()

                # Respawn the player
                self._create_player()

            # Remove the leftmost heart
            self.hearts.sprites()[-1].kill()

        obs = self._get_obs()

        # Create a placeholder for additional information
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, info

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

        if (
            self.player.last_shoot == 0
            or self.steps - self.player.last_shoot >= self.metadata["render_fps"] * 1
        ):
            self.player.last_shoot = self.steps
            player_bullet = self.player.bullet(
                window_width=self.window_width,
                window_height=self.window_height,
                tank_size=self.player.surf.get_size(),
                tank_center=self.player.rect.center,
                angle=angle,
                speed=self.player_speed + self._fps_to_speed(3),
            )
            self.player_bullets.add(player_bullet)
            self.all_sprites.add(player_bullet)

    def _enemy_shoot(self, enemy: Enemy, angle: int, interval: int) -> None:
        """
        An internal function that controls how an enemy shoots. 
        It shoots with an interval of not less than 2 seconds 
        and a probability of 5% (based on a framerate on 30).
        """

        if (
            len(self.enemy_bullets) < self.max_enemy_bullets
            and self.steps - enemy.last_shoot >= self.metadata["render_fps"] * interval
            and self.np_random.random() < self._fps_to_prob(0.05)
        ):
            enemy.last_shoot = self.steps
            enemy_bullet = enemy.bullet(
                window_width=self.window_width,
                window_height=self.window_height,
                tank_size=enemy.surf.get_size(),
                tank_center=enemy.rect.center,
                angle=angle,
                speed=enemy.speed + self._fps_to_speed(2),
            )
            self.enemy_bullets.add(enemy_bullet)
            self.all_sprites.add(enemy_bullet)

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        pygame.init()
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Tank War")
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Garamond", 25)

        # Create a surface to hold all elements
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # Set the background
        bg = pygame.image.load("./images/background.png")
        bg.set_alpha(200)
        canvas.blit(bg, (0, 0))

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

        # Display the player's gun's remaining reloading time as a shrinking rectangle
        if self.player.last_shoot != 0:
            reload_bar_len = max(
                0,
                80 * (self.metadata["render_fps"] * 1 - (self.steps - self.player.last_shoot)) // self.metadata["render_fps"],
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
            # Draw the canvas to the PyGame window
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
            pygame.quit()
