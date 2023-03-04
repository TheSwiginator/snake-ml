from .game_obj import GameObject
from .utils import screen
from .constants import *
import random, pygame

# Game object for storing positional data for food on the board
class Food(GameObject):
    # Total number of Food instances
    _count = 0

    def __init__(self, x, y):
        # Inherited Instance Variables from GameObject
        super().__init__(x, y)
        # Increment total number of Food objects by 1
        Food._count += 1
        # Unique Food object identifier
        self.id = Food._count 
    
    # String representation
    def __str__(self):
        return f"Food: ({self.x}, {self.y})"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()

    # Returns an Food object with randomized x and y values
    @classmethod
    def _spawn(cls):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        return cls(x, y)
        
    # Draws visual representation of this Food object to the running pygame window
    def draw(self):
        # Draw rect to screen
        if self.id == Food._count:
            pygame.draw.rect(screen, (255, 0, 0), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
        else:
            pygame.draw.rect(screen, (80, 80, 80), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))