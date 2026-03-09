import random
import sys
import pygame


# ----------------------------
# Config
# ----------------------------
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 32
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 12

EMPTY = 0
AGENT = 1
FOOD1 = 2
FOOD2 = 3

COLORS = {
    EMPTY: (30, 30, 30),       # dark gray
    AGENT: (255, 255, 255),     # blue
    FOOD1: (0, 255, 0),     # green
    FOOD2: (255, 0, 0),    # purple
    "GRID": (70, 70, 70),
    "BG": (15, 15, 15),
}


# ----------------------------
# GridWorld
# ----------------------------
class GridWorld:
    def __init__(self, width=20, height=20, num_food1=20, num_food2=12):
        self.width = width
        self.height = height
        self.num_food1 = num_food1
        self.num_food2 = num_food2

        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.agent_pos = (0, 0)

        self.reset()

    def reset(self):
        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]

        # Place agent
        self.agent_pos = self._random_empty_cell()
        ax, ay = self.agent_pos
        self.grid[ay][ax] = AGENT

        # Place food1
        for _ in range(self.num_food1):
            x, y = self._random_empty_cell()
            self.grid[y][x] = FOOD1

        # Place food2
        for _ in range(self.num_food2):
            x, y = self._random_empty_cell()
            self.grid[y][x] = FOOD2

    def _random_empty_cell(self):
        while True:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if self.grid[y][x] == EMPTY:
                return x, y

    def move_agent(self, dx, dy):
        ax, ay = self.agent_pos
        nx = ax + dx
        ny = ay + dy

        # Stay in bounds
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return

        target_tile = self.grid[ny][nx]

        # Clear current agent position
        self.grid[ay][ax] = EMPTY

        # Move agent onto target cell
        # For now, stepping on food just replaces it
        self.agent_pos = (nx, ny)
        self.grid[ny][nx] = AGENT

    def draw(self, screen):
        screen.fill(COLORS["BG"])

        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y][x]
                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )

                pygame.draw.rect(screen, COLORS[tile], rect)
                pygame.draw.rect(screen, COLORS["GRID"], rect, 1)


# ----------------------------
# Main loop
# ----------------------------
def main():
    pygame.init()
    pygame.display.set_caption("20x20 GridWorld")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    world = GridWorld(width=20, height=20, num_food1=20, num_food2=12)

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    world.reset()
                elif event.key == pygame.K_UP:
                    world.move_agent(0, -1)
                elif event.key == pygame.K_DOWN:
                    world.move_agent(0, 1)
                elif event.key == pygame.K_LEFT:
                    world.move_agent(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    world.move_agent(1, 0)

        world.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()