import pygame
from environments.simulation import SimulationEnvironment

# Constants
WIDTH, HEIGHT = 800, 600
BALL_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (0, 0, 0)
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Bouncing Ball Simulation")
    clock = pygame.time.Clock()

    env = SimulationEnvironment(WIDTH, HEIGHT, num_balls=1)

    running = True
    while running:
        env.run(steps=1, delay=1/FPS, screen=screen, background_color=BACKGROUND_COLOR, ball_color=BALL_COLOR)
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()