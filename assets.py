import random

import pygame


class Player(pygame.sprite.Sprite):
    resize_ratio = 7
    speed = 3

    def __init__(self, window_width: int, window_height: int, seed: None | int) -> None:
        super().__init__()
        self.seed = seed
        random.seed(self.seed)

        self.window_width = window_width
        self.window_height = window_height

        self.surf = pygame.image.load("images/tank_01/tank_01_A.png").convert()
        self.surf.set_alpha(256)
        self.surf = pygame.transform.scale(self.surf, (self.surf.get_width() / self.resize_ratio, self.surf.get_height() / self.resize_ratio))

        self.angle = random.choice((0, 90, 180, 270))
        self.surf = pygame.transform.rotate(self.surf, self.angle)
        # self.surf.set_colorkey((255, 255, 255), pygame.RLEACCEL)
        self.rect = self.surf.get_rect(
            center=(
                random.randint(self.window_width * 0.2, self.window_width * 0.8),
                random.randint(self.window_height * 0.2, self.window_height * 0.8),
            )
        )
        self._keep_inside()

    def _keep_inside(self):
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


class Enemy(pygame.sprite.Sprite):
    resize_ratio = 5.5
    speed = 2 

    def __init__(self, window_width: int, window_height: int, seed: None | int) -> None:
        super().__init__()
        self.seed = seed
        random.seed(self.seed)

        self.window_width = window_width
        self.window_height = window_height

        self.surf = pygame.image.load("images/tank_02/tank_02_A.png").convert()
        self.surf.set_alpha(256)
        self.surf = pygame.transform.scale(self.surf, (self.surf.get_width() / self.resize_ratio, self.surf.get_height() / self.resize_ratio))

        self.angle = random.choice((0, 90, 180, 270))
        self.surf = pygame.transform.rotate(self.surf, self.angle)
        # self.surf.set_colorkey((255, 255, 255), pygame.RLEACCEL)

        self.rect = self.surf.get_rect(
            center=(
                random.randint(self.window_width * 0.1, self.window_width * 0.9),
                random.randint(self.window_height * 0.1, self.window_height * 0.9),
            )
        )

        # if self.angle == 0:
        #     self.rect = self.surf.get_rect(
        #         center=(
        #             random.randint(self.window_width * 0.1, self.window_width * 0.9),
        #             random.randint(self.window_height * 0.1, self.window_height * 0.9),
        #         )
        #     )

        # elif self.angle == 90:
        #     pass

        # elif self.angle == 180:
        #     pass

        # elif self.angle == 270:
        #     pass
