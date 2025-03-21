import pygame
import random

# --- Constants ---
WIDTH, HEIGHT = 800, 600
NUM_BLOBS = 30
BLOB_RADIUS = 10
FPS = 60
UI_WIDTH = 200  # Width of the UI panel

# Colors for factions
FACTION_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0) # Yellow
]

# Pygame fonts
pygame.init()
FONT = pygame.font.Font(None, 24)

# --- Faction Class ---
class Faction:
    def __init__(self, name, color, attraction, aggression):
        self.name = name
        self.color = color
        self.population = 0  # Tracks the number of blobs in this faction
        self.attraction = attraction  # Determines how strongly members attract each other
        self.aggression = aggression  # Determines repulsion from other factions

# Relationship distance threshold (for drawing lines and computing forces)
RELATIONSHIP_DISTANCE = 120 ** 2  # Squared distance for efficiency

# Force factors
DAMPING = 0.98  # Slight velocity damping to keep movement stable
MIN_FORCE_DISTANCE = 25 ** 2  # Prevents extreme attraction at very close range

class Blob:
    def __init__(self, x, y, faction):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-1.5, 1.5)
        self.faction = faction  # Reference to a faction object
        faction.population += 1  # Increase faction population count

    def move(self):
        self.x += self.vx
        self.y += self.vy

        # Bounce off walls (excluding UI panel area)
        if self.x < BLOB_RADIUS or self.x > WIDTH - UI_WIDTH - BLOB_RADIUS:
            self.vx *= -1
        if self.y < BLOB_RADIUS or self.y > HEIGHT - BLOB_RADIUS:
            self.vy *= -1

    def apply_forces(self, others):
        net_fx, net_fy = 0, 0

        for other in others:
            if other == self:
                continue
            
            dx = other.x - self.x
            dy = other.y - self.y
            squared_dist = dx ** 2 + dy ** 2
            
            if squared_dist < RELATIONSHIP_DISTANCE and squared_dist > 0:
                norm = (squared_dist ** 0.5) if squared_dist > 0 else 1
                
                if squared_dist < MIN_FORCE_DISTANCE:
                    force_factor = 0  # Prevent excessive attraction at very close range
                else:
                    force_factor = self.faction.attraction * (1 - squared_dist / RELATIONSHIP_DISTANCE)
                    
                # Repulsion between different factions based on aggression level
                if self.faction != other.faction:
                    force_factor -= self.faction.aggression * 0.02  # Mild repulsion from different factions
                
                # Apply force scaled by inverse squared distance to prevent orbital motion
                net_fx += force_factor * dx / norm
                net_fy += force_factor * dy / norm

        # Apply calculated forces with damping
        self.vx = (self.vx + net_fx) * DAMPING
        self.vy = (self.vy + net_fy) * DAMPING

    def draw(self, screen):
        pygame.draw.circle(screen, self.faction.color, (int(self.x), int(self.y)), BLOB_RADIUS)

# --- UI Panel ---
def draw_ui(screen, factions):
    pygame.draw.rect(screen, (50, 50, 50), (WIDTH - UI_WIDTH, 0, UI_WIDTH, HEIGHT))
    y_offset = 20
    
    for faction in factions:
        text_surface = FONT.render(f"{faction.name}: {faction.population} blobs", True, (255, 255, 255))
        screen.blit(text_surface, (WIDTH - UI_WIDTH + 10, y_offset))
        y_offset += 30

# --- Main Simulation ---
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Swarm Simulation with Factions UI")
    clock = pygame.time.Clock()

    # Create factions with different attraction and aggression levels
    factions = [
        Faction("Red Faction", (255, 0, 0), attraction=0.02, aggression=0.1),
        Faction("Green Faction", (0, 255, 0), attraction=0.015, aggression=0.05),
        Faction("Blue Faction", (0, 0, 255), attraction=0.01, aggression=0.02),
        Faction("Yellow Faction", (255, 255, 0), attraction=0.025, aggression=0.08)
    ]
    
    # Create blobs with random factions
    blobs = [
        Blob(random.randint(BLOB_RADIUS, WIDTH - UI_WIDTH - BLOB_RADIUS),
             random.randint(BLOB_RADIUS, HEIGHT - BLOB_RADIUS),
             random.choice(factions))
        for _ in range(NUM_BLOBS)
    ]

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Reset faction populations for recalculation
        for faction in factions:
            faction.population = 0
        
        # Apply forces and move blobs
        for blob in blobs:
            blob.apply_forces(blobs)
            blob.move()
            blob.faction.population += 1

        # Draw everything
        screen.fill((30, 30, 30))
        for blob in blobs:
            blob.draw(screen)
        
        # Draw UI
        draw_ui(screen, factions)
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
