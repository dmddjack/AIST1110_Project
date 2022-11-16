import random

import pygame


class _Tank(pygame.sprite.Sprite):
    def __init__(self, window_width: int, window_height: int, image_path: str, speed: int, resize_ratio: int = 1, seed: None | int = None):
        super().__init__()
        # Seed the random module
        random.seed(seed)

        self.window_width, self.window_height = window_width, window_height
        self.speed = speed

        self.surf = pygame.image.load(image_path)
        self.surf.set_alpha(256)
        self.surf = pygame.transform.scale(self.surf, (self.surf.get_width() / resize_ratio, self.surf.get_height() / resize_ratio))

        self.angle = random.choice((0, 90, 180, 270))
        self.surf = pygame.transform.rotate(self.surf, self.angle)

    def _keep_inside(self) -> None:
        if self.rect.left < 0:
            self.rect.left = 0
        elif self.rect.right > self.window_width:
            self.rect.right = self.window_width
        if self.rect.top <= 0:
            self.rect.top = 0
        elif self.rect.bottom >= self.window_height:
            self.rect.bottom = self.window_height

    def update(self, d_x, d_y, angle: int) -> None:
        self.surf = pygame.transform.rotate(self.surf, self._angle2rotation(angle))
        self.angle = angle
        self.rect = self.surf.get_rect(center=self.rect.center)
        self.rect.move_ip(d_x * self.speed, d_y * self.speed)
        self._keep_inside()

    def _angle2rotation(self, angle: int) -> int:
        rotation = angle - self.angle
        if rotation < 0:
            rotation = 360 + rotation
        return rotation


class Player(_Tank):
    image_path = "images/tank_01/tank_01_A.png"
    resize_ratio = 5.5

    def __init__(self, window_width: int, window_height: int, speed: int, seed: None | int = None) -> None:
        super().__init__(window_width, window_height, self.image_path, speed, self.resize_ratio, seed)
        self.rect = self.surf.get_rect(
            center=(
                random.randint(self.window_width * 0.2, self.window_width * 0.8),
                random.randint(self.window_height * 0.2, self.window_height * 0.8),
            )
        )
        self._keep_inside()


class Enemy(_Tank):
    image_path = "images/tank_02/tank_02_A.png"
    resize_ratio = 4.4

    def __init__(self, window_width: int, window_height: int, speed: int, seed: None | int = None) -> None:
        super().__init__(window_width, window_height, self.image_path, speed, self.resize_ratio, seed)
        if self.angle == 0:
            self.start_x = random.randint(0, self.window_width)
            self.start_y = self.window_height
        elif self.angle == 90:
            self.start_x = self.window_width
            self.start_y = random.randint(0, self.window_height)
        elif self.angle == 180:
            self.start_x = random.randint(0, self.window_width)
            self.start_y = 0
        else:
            self.start_x = 0
            self.start_y = random.randint(0, self.window_height)
        self.rect = self.surf.get_rect(center=(self.start_x, self.start_y))
        self._keep_inside()
