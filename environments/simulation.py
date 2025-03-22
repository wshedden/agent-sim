import time
import random

class Ball:
    def __init__(self, x, y, vx, vy, radius=20):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius

    def move(self):
        self.x += self.vx
        self.y += self.vy

    def bounce(self, width, height):
        if self.x - self.radius < 0 or self.x + self.radius > width:
            self.vx = -self.vx
        if self.y - self.radius < 0 or self.y + self.radius > height:
            self.vy = -self.vy

    def draw(self, screen, color):
        import pygame
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)

    def __str__(self):
        return f"Ball at ({self.x}, {self.y}) with velocity ({self.vx}, {self.vy})"

class SimulationEnvironment:
    def __init__(self, width, height, num_balls):
        self.width = width
        self.height = height
        self.balls = [self.create_random_ball() for _ in range(num_balls)]

    def create_random_ball(self):
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        vx = random.uniform(-5, 5)
        vy = random.uniform(-5, 5)
        return Ball(x, y, vx, vy)

    def run(self, steps, delay, screen, background_color, ball_color):
        import pygame
        for step in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            for ball in self.balls:
                ball.move()
                ball.bounce(self.width, self.height)

            screen.fill(background_color)
            for ball in self.balls:
                ball.draw(screen, ball_color)
            pygame.display.flip()

            time.sleep(delay)