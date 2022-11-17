import pygame


class _Bullet(pygame.sprite.Sprite):
    def __init__(self, tank_size: tuple[int, int], tank_center: tuple[int, int], angle: int, speed: int, image_path: str, resize_ratio: int = 1) -> None:
        super().__init__()

        self.angle = angle
        self.speed = speed

        self.surf = pygame.image.load(image_path)
        self.surf.set_alpha(256)
        self.surf = pygame.transform.scale(self.surf, (self.surf.get_width() / resize_ratio, self.surf.get_height() / resize_ratio))
        self.surf = pygame.transform.rotate(self.surf, self.angle)
        if self.angle == 0:
            start_x = tank_center[0] + 1
            start_y = tank_center[1] - tank_size[1] // 2 - self.surf.get_height() // 2
        elif self.angle == 90:
            start_x = tank_center[0] - tank_size[0] // 2 - self.surf.get_width() // 2
            start_y = tank_center[1]
        elif self.angle == 180:
            start_x = tank_center[0]
            start_y = tank_center[1] + tank_size[1] // 2 + self.surf.get_height() // 2
        else:
            start_x = tank_center[0] + tank_size[0] // 2 + self.surf.get_width() // 2
            start_y = tank_center[1]
        self.rect = self.surf.get_rect(center=(start_x, start_y))

    def move(self) -> None:
        if self.angle == 0:
            self.rect.move_ip(0, -self.speed)
        elif self.angle == 90:
            self.rect.move_ip(-self.speed, 0)
        elif self.angle == 180:
            self.rect.move_ip(0, self.speed)
        else:
            self.rect.move_ip(self.speed, 0)


class _PlayerBullet(_Bullet):
    image_path = "images/bullets/bullet_01.png"
    resize_ratio = 2

    def __init__(self, tank_size: tuple[int, int], tank_center: tuple[int, int], angle: int, speed: int) -> None:
        super().__init__(tank_size, tank_center, angle, speed, self.image_path, self.resize_ratio)


class _EnemyBullet(_Bullet):
    image_path = "images/bullets/bullet_02.png"
    resize_ratio = 2
    
    def __init__(self, tank_size: tuple[int, int], tank_center: tuple[int, int], angle: int, speed: int) -> None:
        super().__init__(tank_size, tank_center, angle, speed, self.image_path, self.resize_ratio)


class _Tank(pygame.sprite.Sprite):
    def __init__(self, bullet: _Bullet, start_x: int, start_y: int, start_angle: int, window_width: int, window_height: int, speed: int, image_path: str, resize_ratio: int = 1) -> None:
        super().__init__()

        self.bullet = bullet
        self.angle = start_angle
        self.window_width, self.window_height = window_width, window_height
        self.speed = speed

        self.surf = pygame.image.load(image_path)
        self.surf.set_alpha(256)
        self.surf = pygame.transform.scale(self.surf, (self.surf.get_width() / resize_ratio, self.surf.get_height() / resize_ratio))
        self.surf = pygame.transform.rotate(self.surf, self.angle)
        self.rect = self.surf.get_rect(center=(start_x, start_y))
        self._keep_inside()

    def _keep_inside(self) -> None:
        if self.rect.left < 0:
            self.rect.left = 0
        elif self.rect.right > self.window_width:
            self.rect.right = self.window_width
        if self.rect.top <= 0:
            self.rect.top = 0
        elif self.rect.bottom >= self.window_height:
            self.rect.bottom = self.window_height

    def update(self, dx, dy, angle: int) -> None:
        self.surf = pygame.transform.rotate(self.surf, self._angle2rotation(angle))
        self.angle = angle
        self.rect = self.surf.get_rect(center=self.rect.center)
        self.rect.move_ip(dx * self.speed, dy * self.speed)
        self._keep_inside()

    def _angle2rotation(self, angle: int) -> int:
        rotation = angle - self.angle
        if rotation < 0:
            rotation = 360 + rotation
        return rotation


class Player(_Tank):
    image_path = "images/tank_01/tank_01_A.png"
    resize_ratio = 5.5

    def __init__(self, start_x: int, start_y: int, start_angle: int, window_width: int, window_height: int, speed: int) -> None:
        super().__init__(_PlayerBullet, start_x, start_y, start_angle, window_width, window_height, speed, self.image_path, self.resize_ratio)


class Enemy(_Tank):
    image_path = "images/tank_02/tank_02_A.png"
    resize_ratio = 4.4

    def __init__(self, start_x: int, start_y: int, start_angle: int, window_width: int, window_height: int, speed: int) -> None:
        super().__init__(_EnemyBullet, start_x, start_y, start_angle, window_width, window_height, speed, self.image_path, self.resize_ratio)
