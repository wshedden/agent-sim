import pygame
import random
import math
from pygame.math import Vector2

# --- Constants ---
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (30, 30, 30)
FPS = 60

FRICTION = 0.98            # Friction slows blobs down over time
MIN_RADIUS = 15
MAX_RADIUS = 30
MESSAGE_DURATION = 120     # Duration (in frames) of a conversation

# Energy management constants
MAX_ENERGY = 100
ENERGY_CONSUMPTION_RATE = 0.2   # Energy consumed per unit speed when moving fast
ENERGY_REGEN_RATE = 0.5         # Energy regained per frame when moving slowly
SPEED_THRESHOLD = 0.5           # Speed below which blob is considered nearly still

# Behavior constants
DETECTION_RADIUS = 300        # Range to look for other blobs to seek
SEEK_FORCE = 0.05             # Acceleration magnitude when seeking another blob
WANDER_FORCE = 0.02           # Random acceleration for wandering
SEPARATION_FORCE = 2          # Gentle separation speed applied after conversation

# Predefined messages for blob interactions
INTERACTION_MESSAGES = [
    "Hello!", "How are you?", "Nice meeting you!", "Interesting!", "Goodbye!"
]

# --- Blob Class ---
class Blob:
    _id_counter = 0

    def __init__(self, x, y, radius=None, color=None):
        self.id = Blob._id_counter
        Blob._id_counter += 1

        self.radius = radius if radius is not None else random.randint(MIN_RADIUS, MAX_RADIUS)
        self.pos = Vector2(x, y)
        self.vel = Vector2(random.uniform(-4, 4), random.uniform(-4, 4))
        self.color = color if color is not None else (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        self.mass = self.radius ** 2
        self.energy = MAX_ENERGY
        self.max_energy = MAX_ENERGY
        self.conversed_with = set()
        self.message = None
        self.message_timer = 0

    def apply_force(self, force):
        """Apply an acceleration (force) to the blob's velocity."""
        self.vel += force

    def move(self):
        # Update position based on current velocity
        self.pos += self.vel
        # Apply friction to velocity
        self.vel *= FRICTION
        # Energy management: consume energy when moving fast; regenerate if moving slowly.
        speed = self.vel.length()
        if speed > SPEED_THRESHOLD:
            self.energy -= ENERGY_CONSUMPTION_RATE * speed
        else:
            self.energy = min(self.max_energy, self.energy + ENERGY_REGEN_RATE)
        if self.energy <= 0:
            self.vel = Vector2(0, 0)

    def check_wall_collision(self, width, height):
        # Reflect from left/right boundaries
        if self.pos.x - self.radius < 0:
            self.pos.x = self.radius
            self.vel.x *= -1
        elif self.pos.x + self.radius > width:
            self.pos.x = width - self.radius
            self.vel.x *= -1
        # Reflect from top/bottom boundaries
        if self.pos.y - self.radius < 0:
            self.pos.y = self.radius
            self.vel.y *= -1
        elif self.pos.y + self.radius > height:
            self.pos.y = height - self.radius
            self.vel.y *= -1

    def draw(self, screen, font):
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)
        if self.message and self.message_timer > 0:
            text_surface = font.render(self.message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.pos.x, self.pos.y - self.radius - 15))
            screen.blit(text_surface, text_rect)

    def update_message(self):
        if self.message_timer > 0:
            self.message_timer -= 1
        if self.message_timer <= 0:
            self.message = None

    def set_message(self, message):
        self.message = message
        self.message_timer = MESSAGE_DURATION

# --- Simulation Environment Class ---
class SimulationEnvironment:
    def __init__(self, width, height, num_blobs=5):
        self.width = width
        self.height = height
        self.blobs = []
        self.conversations = {}  # Track ongoing conversations: key = frozenset({id1, id2})
        for _ in range(num_blobs):
            x = random.randint(MAX_RADIUS, width - MAX_RADIUS)
            y = random.randint(MAX_RADIUS, height - MAX_RADIUS)
            self.blobs.append(Blob(x, y))

    def add_blob(self, x, y):
        self.blobs.append(Blob(x, y))

    def apply_behaviors(self):
        for blob in self.blobs:
            # Skip blobs in conversation
            if blob.message is not None:
                continue

            # Always apply a wander force for random walking
            wander_force = Vector2(random.uniform(-WANDER_FORCE, WANDER_FORCE),
                                   random.uniform(-WANDER_FORCE, WANDER_FORCE))
            blob.apply_force(wander_force)
            
            # With a low probability, also try to seek a nearby blob.
            if random.random() < 0.1:  # 10% chance to seek target
                target = None
                min_distance = float('inf')
                for other in self.blobs:
                    if other.id == blob.id or other.message is not None or other.id in blob.conversed_with:
                        continue
                    distance = (other.pos - blob.pos).length()
                    if distance < min_distance and distance < DETECTION_RADIUS:
                        min_distance = distance
                        target = other
                if target and blob.energy > 20:
                    desired = (target.pos - blob.pos).normalize() * 4  # 4 is arbitrary max speed
                    steering = desired - blob.vel
                    blob.apply_force(steering * SEEK_FORCE)

    def handle_interactions(self):
        n = len(self.blobs)
        for i in range(n):
            for j in range(i + 1, n):
                blob1 = self.blobs[i]
                blob2 = self.blobs[j]
                if blob2.id in blob1.conversed_with:
                    continue
                if blob1.message is not None or blob2.message is not None:
                    continue

                distance = (blob2.pos - blob1.pos).length()
                if distance < blob1.radius + blob2.radius:
                    pair_key = frozenset({blob1.id, blob2.id})
                    if pair_key not in self.conversations:
                        self.conversations[pair_key] = MESSAGE_DURATION
                        msg = random.choice(INTERACTION_MESSAGES)
                        blob1.set_message(msg)
                        blob2.set_message(msg)
                        blob1.vel = Vector2(0, 0)
                        blob2.vel = Vector2(0, 0)

    def update_conversations(self):
        finished = []
        for pair_key in list(self.conversations.keys()):
            self.conversations[pair_key] -= 1
            if self.conversations[pair_key] <= 0:
                finished.append(pair_key)
        for pair_key in finished:
            id1, id2 = tuple(pair_key)
            blob1 = next(blob for blob in self.blobs if blob.id == id1)
            blob2 = next(blob for blob in self.blobs if blob.id == id2)
            # Apply gentle separation
            separation_dir = (blob2.pos - blob1.pos).normalize()
            blob1.vel = -separation_dir * SEPARATION_FORCE
            blob2.vel = separation_dir * SEPARATION_FORCE
            blob1.message = None
            blob2.message = None
            blob1.conversed_with.add(blob2.id)
            blob2.conversed_with.add(blob1.id)
            del self.conversations[pair_key]

    def update(self):
        self.apply_behaviors()
        for blob in self.blobs:
            blob.move()
            blob.check_wall_collision(self.width, self.height)
            blob.update_message()
        self.handle_interactions()
        self.update_conversations()

    def draw(self, screen):
        font = pygame.font.SysFont("Arial", 16)
        for blob in self.blobs:
            blob.draw(screen, font)
        info_font = pygame.font.SysFont("Arial", 20)
        info_surface = info_font.render(f"Ongoing Conversations: {len(self.conversations)}", True, (255, 255, 255))
        screen.blit(info_surface, (10, 10))

    def run_step(self, screen, background_color):
        self.update()
        screen.fill(background_color)
        self.draw(screen)

# --- Main Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Improved Interacting Blobs Simulation")
    clock = pygame.time.Clock()
    env = SimulationEnvironment(WIDTH, HEIGHT, num_blobs=5)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                env.add_blob(x, y)
        env.run_step(screen, BACKGROUND_COLOR)
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()

if __name__ == '__main__':
    main()
