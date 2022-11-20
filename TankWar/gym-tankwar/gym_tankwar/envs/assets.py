#!/usr/bin/env python3

import numpy as np
import pygame

CONT_OBS_SPACE = True  # For testing purposes

class _Movable(pygame.sprite.Sprite):
    def __init__(
            self,
            window_width: int,
            window_height: int,
            start_x: int,
            start_y: int,
            start_angle: int,
            speed: int,
            image_path: str,
            resize_ratio: float = 1.0) -> None:
        super().__init__()

        self.window_width = window_width
        self.window_height = window_height
        self.angle = start_angle
        self.speed = speed

        # Load the image and convert it into a Surface
        self.surf = pygame.image.load(image_path)

        # Set the background to transparent
        self.surf.set_alpha(256)

        # Resize the image
        self.surf = pygame.transform.scale(
            self.surf,
            (
                self.surf.get_width() / resize_ratio,
                self.surf.get_height() / resize_ratio,
            ),
        )

        # Rotate the image
        self.surf = pygame.transform.rotate(self.surf, self.angle)

        # Get the Rect of the Surface
        self.rect = self.surf.get_rect(center=(start_x, start_y))

    def get_obs(self) -> np.ndarray:
        """A function that returns the movable's location."""

        if CONT_OBS_SPACE:
            return np.array(
                (
                    self.rect.center[0] / self.window_width, 
                    self.rect.center[1] / self.window_height, 
                    self.angle / 360, 
                ),
                dtype=np.float32,
            )
        else:
            return np.array(
                (self.rect.center[0], self.rect.center[1], self.angle),
                dtype=int,
            )


# We need to create class _Bullet before class _Tank 
# because the latter one uses the former one.
class _Bullet(_Movable):
    def __init__(
            self,
            window_width: int,
            window_height: int,
            tank_size: tuple[int, int],
            tank_center: tuple[int, int],
            angle: int,
            speed: int,
            image_path: str,
            resize_ratio: float = 1.0) -> None:
        # Initialize _Movable with (0, 0) as the starting location first
        super().__init__(
            window_width=window_width,
            window_height=window_height,
            start_x=0,
            start_y=0,
            start_angle=angle,
            speed=speed,
            image_path=image_path,
            resize_ratio=resize_ratio
        )

        # Determine the starting location of the bullet based on 
        # the location and the angle of the tank
        if angle == 0:
            start_x = tank_center[0] + 1  # The "+1" is for pixel adjustment
            start_y = tank_center[1] - tank_size[1] // 2 \
                - self.surf.get_height() // 2
        elif angle == 90:
            start_x = tank_center[0] - tank_size[0] // 2 \
                - self.surf.get_width() // 2
            start_y = tank_center[1]
        elif angle == 180:
            start_x = tank_center[0]
            start_y = tank_center[1] + tank_size[1] // 2 \
                + self.surf.get_height() // 2
        else:
            start_x = tank_center[0] + tank_size[0] // 2 \
                + self.surf.get_width() // 2
            start_y = tank_center[1]

        # Utilize the calculated starting location
        self.rect = self.surf.get_rect(center=(start_x, start_y))

    def move(self) -> None:
        """A function that moves the bullet."""

        if self.angle == 0:
            self.rect.move_ip(0, -self.speed)
        elif self.angle == 90:
            self.rect.move_ip(-self.speed, 0)
        elif self.angle == 180:
            self.rect.move_ip(0, self.speed)
        else:
            self.rect.move_ip(self.speed, 0)


class _PlayerBullet(_Bullet):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = "./images/bullets/bullet_01.png"

    resize_ratio = 2

    def __init__(
            self,
            window_width: int,
            window_height: int,
            tank_size: tuple[int, int],
            tank_center: tuple[int, int],
            angle: int,
            speed: int) -> None:
        super().__init__(
            window_width=window_width,
            window_height=window_height,
            tank_size=tank_size,
            tank_center=tank_center,
            angle=angle,
            speed=speed,
            image_path=self.image_path,
            resize_ratio=self.resize_ratio,
        )


class _EnemyBullet(_Bullet):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = "./images/bullets/bullet_02.png"

    resize_ratio = 1.5

    def __init__(
            self,
            window_width: int,
            window_height: int,
            tank_size: tuple[int, int],
            tank_center: tuple[int, int],
            angle: int,
            speed: int) -> None:
        super().__init__(
            window_width=window_width,
            window_height=window_height,
            tank_size=tank_size,
            tank_center=tank_center,
            angle=angle,
            speed=speed,
            image_path=self.image_path,
            resize_ratio=self.resize_ratio,
        )


class _Tank(_Movable):
    def __init__(
            self,
            window_width: int,
            window_height: int,
            bullet: _Bullet,
            start_x: int,
            start_y: int,
            start_angle: int,
            speed: int,
            last_shoot: int,
            image_path: str,
            resize_ratio: float = 1.0) -> None:
        super().__init__(
            window_width=window_width,
            window_height=window_height,
            start_x=start_x,
            start_y=start_y,
            start_angle=start_angle,
            speed=speed,
            image_path=image_path,
            resize_ratio=resize_ratio,
        )

        # Keep the tank inside the window
        self._keep_inside()

        # Store the type of bullet used by the tank, 
        # either _PlayerBullet or _EnemyBullet
        self.bullet = bullet

        # Store the last step when the tank so the next step when the tank 
        # can shoot can be calculated
        self.last_shoot = last_shoot

    def _keep_inside(self) -> tuple[bool, list[int]]:
        """An internal function that keeps the tank inside the window."""

        touches_border = False
        correction_angles = []
        if self.rect.left < 0:
            self.rect.left = 0
            touches_border = True
            correction_angles.append(270)
        elif self.rect.right > self.window_width:
            self.rect.right = self.window_width
            touches_border = True
            correction_angles.append(90)
        if self.rect.top < 0:
            self.rect.top = 0
            touches_border = True
            correction_angles.append(180)
        elif self.rect.bottom > self.window_height:
            self.rect.bottom = self.window_height
            touches_border = True
            correction_angles.append(0)

        return touches_border, correction_angles

    def update(self, dx: int, dy: int, 
            new_angle: int) -> tuple[bool, list[int]]:
        """
        A function that rotates the tank (if necessary) and 
        moves the tank
        """

        # Rotate the Surface if necessary
        if new_angle != self.angle:
            self.surf = pygame.transform.rotate(
                self.surf, self._angle_to_rotation(new_angle)
            )
            self.rect = self.surf.get_rect(center=self.rect.center)
            self.angle = new_angle

        # Move the Rect
        self.rect.move_ip(dx * self.speed, dy * self.speed)

        touches_border, correction_angles = self._keep_inside()

        return touches_border, correction_angles

    def _angle_to_rotation(self, target_angle: int) -> int:
        """
        An internal function that maps the target angle to 
        a suitable rotation.
        """

        rotation = target_angle - self.angle
        if rotation < 0:
            rotation = 360 + rotation

        return rotation


class Player(_Tank):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = "./images/tank_01/tank_01_A.png"

    resize_ratio = 5.5

    def __init__(
            self,
            window_width: int,
            window_height: int,
            start_x: int,
            start_y: int,
            start_angle: int,
            speed: int) -> None:
        super().__init__(
            window_width=window_width,
            window_height=window_height,
            bullet=_PlayerBullet,
            start_x=start_x,
            start_y=start_y,
            start_angle=start_angle,
            speed=speed,
            last_shoot=0,
            image_path=self.image_path,
            resize_ratio=self.resize_ratio,
        )


class Enemy(_Tank):
    # Image source: https://craftpix.net/freebies/free-2d-battle-tank-game-assets/
    # License: https://craftpix.net/file-licenses/
    image_path = "./images/tank_02/tank_02_A.png"

    resize_ratio = 4.4

    def __init__(
            self,
            window_width: int,
            window_height: int,
            start_x: int,
            start_y: int,
            start_angle: int,
            speed: int,
            creation_step: int) -> None:
        super().__init__(
            window_width=window_width,
            window_height=window_height,
            bullet=_EnemyBullet,
            start_x=start_x,
            start_y=start_y,
            start_angle=start_angle,
            speed=speed,
            last_shoot=creation_step,
            image_path=self.image_path,
            resize_ratio=self.resize_ratio,
        )
        self.last_rotate = creation_step


class Heart(pygame.sprite.Sprite):
    # Image source: https://opengameart.org/content/heart-1
    # License: https://creativecommons.org/publicdomain/zero/1.0/
    image_path = "./images/heart.png"

    resize_ratio = 5

    def __init__(self, window_width: int, order: int) -> None:
        super().__init__()
        self.surf = pygame.image.load(self.image_path)
        self.surf.set_alpha(256)
        self.surf = pygame.transform.scale(
            self.surf,
            (
                self.surf.get_width() / self.resize_ratio,
                self.surf.get_height() / self.resize_ratio,
            ),
        )
        self.rect = self.surf.get_rect(
            center=(
                window_width - order * self.surf.get_width() + 7,
                self.surf.get_height() // 2 + 5,
            )
        )
