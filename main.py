import pygame
import random
import math

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
        self.x = x
        self.y = y
        # Initial random velocities (top-down, no gravity)
        self.vx = random.uniform(-4, 4)
        self.vy = random.uniform(-4, 4)
        self.color = color if color is not None else (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
        self.mass = self.radius ** 2
        self.energy = MAX_ENERGY
        self.max_energy = MAX_ENERGY
        # Set to track blobs with which this blob has conversed (and will no longer seek)
        self.conversed_with = set()
        # Active conversation message and timer (None if not conversing)
        self.message = None
        self.message_timer = 0

    def move(self):
        # Update position based solely on velocity (top-down physics)
        self.x += self.vx
        self.y += self.vy

        # Apply friction
        self.vx *= FRICTION
        self.vy *= FRICTION

        # Energy management: consume energy when moving fast; regenerate if moving slowly.
        speed = math.hypot(self.vx, self.vy)
        if speed > SPEED_THRESHOLD:
            self.energy -= ENERGY_CONSUMPTION_RATE * speed
        else:
            self.energy = min(self.max_energy, self.energy + ENERGY_REGEN_RATE)
        if self.energy <= 0:
            self.vx, self.vy = 0, 0

    def check_wall_collision(self, width, height):
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx *= -1
        elif self.x + self.radius > width:
            self.x = width - self.radius
            self.vx *= -1
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy *= -1
        elif self.y + self.radius > height:
            self.y = height - self.radius
            self.vy *= -1

    def draw(self, screen, font):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        if self.message and self.message_timer > 0:
            text_surface = font.render(self.message, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(self.x, self.y - self.radius - 15))
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
        # Track ongoing conversations: key is frozenset({id1, id2}), value is remaining frames
        self.conversations = {}
        for _ in range(num_blobs):
            x = random.randint(MAX_RADIUS, width - MAX_RADIUS)
            y = random.randint(MAX_RADIUS, height - MAX_RADIUS)
            self.blobs.append(Blob(x, y))

    def add_blob(self, x, y):
        self.blobs.append(Blob(x, y))

    def apply_behaviors(self):
        # For each blob not in conversation, apply wander and seek behaviors.
        for blob in self.blobs:
            if blob.message is not None:
                continue  # Skip blobs in conversation
            # Wander: add a small random acceleration.
            blob.vx += random.uniform(-WANDER_FORCE, WANDER_FORCE)
            blob.vy += random.uniform(-WANDER_FORCE, WANDER_FORCE)
            
            # Seek: find the nearest free blob (not in conversation and not already conversed with)
            target = None
            min_distance = float('inf')
            for other in self.blobs:
                if other.id == blob.id:
                    continue
                if other.message is not None:
                    continue
                if other.id in blob.conversed_with:
                    continue
                dx = other.x - blob.x
                dy = other.y - blob.y
                distance = math.hypot(dx, dy)
                if distance < min_distance and distance < DETECTION_RADIUS:
                    min_distance = distance
                    target = other
            # If a target is found and blob has energy, accelerate towards it.
            if target and blob.energy > 20:
                dx = target.x - blob.x
                dy = target.y - blob.y
                distance = math.hypot(dx, dy) or 1.0
                # Calculate gentle seek acceleration
                ax = SEEK_FORCE * dx / distance
                ay = SEEK_FORCE * dy / distance
                blob.vx += ax
                blob.vy += ay

    def handle_interactions(self):
        n = len(self.blobs)
        for i in range(n):
            for j in range(i + 1, n):
                blob1 = self.blobs[i]
                blob2 = self.blobs[j]
                # Skip if already conversed
                if blob2.id in blob1.conversed_with:
                    continue
                # Check if either is already in conversation
                if blob1.message is not None or blob2.message is not None:
                    continue

                dx = blob2.x - blob1.x
                dy = blob2.y - blob1.y
                distance = math.hypot(dx, dy)
                if distance < blob1.radius + blob2.radius:
                    pair_key = frozenset({blob1.id, blob2.id})
                    if pair_key not in self.conversations:
                        # Start conversation: stop blobs and assign a message
                        self.conversations[pair_key] = MESSAGE_DURATION
                        msg = random.choice(INTERACTION_MESSAGES)
                        blob1.set_message(msg)
                        blob2.set_message(msg)
                        blob1.vx, blob1.vy = 0, 0
                        blob2.vx, blob2.vy = 0, 0

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
            # Gentle separation: apply a modest separation velocity.
            dx = blob2.x - blob1.x
            dy = blob2.y - blob1.y
            distance = math.hypot(dx, dy) or 1.0
            nx, ny = dx / distance, dy / distance
            blob1.vx = -SEPARATION_FORCE * nx
            blob1.vy = -SEPARATION_FORCE * ny
            blob2.vx = SEPARATION_FORCE * nx
            blob2.vy = SEPARATION_FORCE * ny
            # Clear conversation messages
            blob1.message = None
            blob2.message = None
            # Mark these blobs as having conversed
            blob1.conversed_with.add(blob2.id)
            blob2.conversed_with.add(blob1.id)
            del self.conversations[pair_key]

    def update(self):
        # Apply seeking and wandering behaviors to free blobs
        self.apply_behaviors()
        # Update movement and wall collisions
        for blob in self.blobs:
            blob.move()
            blob.check_wall_collision(self.width, self.height)
            blob.update_message()
        # Handle interactions (collision/conversation initiation)
        self.handle_interactions()
        # Update conversation timers and separate blobs after conversation
        self.update_conversations()

    def draw(self, screen):
        font = pygame.font.SysFont("Arial", 16)
        for blob in self.blobs:
            blob.draw(screen, font)
        info_font = pygame.font.SysFont("Arial", 20)
        info_surface = info_font.render(f"Ongoing Conversations: {len(self.conversations)}", True, (255, 255, 255))
        screen.blit(info_surface, (10, 10))
        # Display energy for each blob
        for blob in self.blobs:
            energy_text = font.render(f"{int(blob.energy)}", True, (255, 255, 0))
            screen.blit(energy_text, (int(blob.x) - blob.radius, int(blob.y) - blob.radius))

    def run_step(self, screen, background_color):
        self.update()
        screen.fill(background_color)
        self.draw(screen)

# --- Main Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Interacting Blobs - Top Down Energy & Seeking Simulation")
    clock = pygame.time.Clock()
    env = SimulationEnvironment(WIDTH, HEIGHT, num_blobs=5)

    running = True
    while running:
        for event in pygame.event.get():
            # Allow exit on window close
            if event.type == pygame.QUIT:
                running = False
            # Add a new blob on mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                env.add_blob(x, y)
                
        env.run_step(screen, BACKGROUND_COLOR)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()
