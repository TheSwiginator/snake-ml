from .game_obj import GameObject
from .food_obj import Food
from .constants import *
from .utils import screen
import random, pygame
import numpy as np
from datetime import datetime

class Snake(GameObject):
    _count = 0 # Total number of Snake instances
    _activeFood = [] # Current food active in the generation
    _startingFood = 6 # THIS IS FOR DEBUGGING PURPOSES ONLY

    def __init__(self, nn, generation, x=int(WIDTH/2), y=int(HEIGHT/2)):
        # Inherited Instance Variables from GameObject #
        super().__init__(x, y)

        # Unique Instance Variables #
        self.neuralNetworkObject = nn # Stores the snakes brain as a NeuralNetwork object
        self.fitness = 1 # Tracks the fitness score of the snake
        self.direction = 0 # Provide extra positional data
        self.bodyCoordinates = [] # History of snake's previous positions up to snake's length (food_eaten), acts as a queue
        self.numberOfFoodEaten = Snake._startingFood # Tracks the number of food eaten upon birth
        self.totalMovesTaken = 1 # Tracks the number of steps made since birth
        self.hungryMovesTaken = 1 # Tracks number of steps made since last meal
        self.dead = False # Tracks if snake is dead or alive
        self.deathDistanceToFood = 0 # Tracks the euclidean distance between the snake and target food upon death
        self.distances = [500] # Tracks the euclidean distance between the snake and target food at each step
        self.positionHistory = [] # Tracks the snake's position at each step
        self.hungryHistory = [] # Tracks the snake's hungry moves at each step since last meal
        self.timeBorn = datetime.timestamp(datetime.now()) # Tracks the time the snake was born
        self.timeDied = None # Tracks the time the snake died
        self.generation = generation # Tracks the generation the snake was born in
        self.startPosition = (x, y) # Tracks the active starting position of the snake
        self.deathPosition = None # Tracks the position the snake died at
        self.autoPilot = False # Tracks whether or not the snake is autopiloted
        self.update() # Intially update the snake's state upon Snake object construction
        Snake._count += 1 # Increment the number of total snakes by 1

    # Compute the snake's next move
    def _think(self, food):
        # Capture positional data in numpy array
        brainInputs = np.array([
            self.x, 
            self.y, 
            food.x, 
            food.y, 
            Snake._inBounds(self.x - 1, self.y), 
            Snake._inBounds(self.x + 1, self.y), 
            Snake._inBounds(self.x, self.y - 1), 
            Snake._inBounds(self.x, self.y + 1),
            (self.x + 1, self.y) in self.bodyCoordinates,
            (self.x - 1, self.y) in self.bodyCoordinates,
            (self.x, self.y + 1) in self.bodyCoordinates,
            (self.x, self.y - 1) in self.bodyCoordinates
        ])

        # Feed the food positional data and snake data into the neural net
        brainOutput = self.neuralNetworkObject.feedForward(brainInputs)
        # Handle neural net output
        computedPrecedenceOfChoices = np.argsort(brainOutput, axis=0)[::-1]
        
        directionsTouchingFood = [GameObject(self.x + i, self.y + j) == self.targetFood for i, j in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        for currentDirection, isCurrentDirectionTouching in enumerate(directionsTouchingFood):
            if isCurrentDirectionTouching:
                self.direction = currentDirection
                return

        
        for currentChoiceIndex, currentChoice in enumerate(computedPrecedenceOfChoices):
            if abs(self.direction - currentChoice) == 2 and self.direction != currentChoice:
                continue

            # Chance for random path change diminishes as you go down the list of options
            if random.random() < 0.7 * (1 / (currentChoiceIndex + 1)) and self.autoPilot:
                try:
                    # Don't randomize path if next option hits body
                    if not [(self.x + i, self.y + j) in self.bodyCoordinates for i, j in [(1, 0), (0, 1), (-1, 0), (0, -1)]][currentChoiceIndex + 1]:
                        continue
                except IndexError:
                    continue

            self.direction = currentChoice
            break
        
        

    
    # Compute the euclidean distance between two GameObjects
    def _getDistanceToFood(self, food):
        return ((self.x - food.x)**2 + (self.y - food.y)**2)**(1/2)
    
    def _getDistanceToFoodOnDeath(self):
        if self.deathPosition is None:
            return 0
        
        initialDistance = ((self.startPosition[0] - self.targetFood.x)**2 + (self.startPosition[1] - self.targetFood.y)**2)**(1/2)
        finalDistance = ((self.deathPosition[0] - self.targetFood.x)**2 + (self.deathPosition[1] - self.targetFood.y)**2)**(1/2)

        return finalDistance - initialDistance
    
    # Returns the snake's score
    def getNumberOfFoodEaten(self):
        '''
        DEBUG NOTE:
        Because we don't want more than 1 food to be present when the starting food score is higher than 1, 
        we must offset the food eaten with our initial food score constant.
        '''
        return self.numberOfFoodEaten - Snake._startingFood
    
    # Returns and sets the fitness score of this snake
    def _calculateFitness(self) -> int:
        return int(400 / (min(self.distances) + 1)) + int(self.getNumberOfFoodEaten()**2 * 20)
    
    def _calculateAverageDistance(self) -> float:
        if len(self.distances) == 0:
            return 0
        return sum(self.distances) / len(self.distances)
    
    # Returns the average number of moves the snake was hungry for
    def calculateTimeToFood(self) -> float:
        if len(self.hungryHistory) == 0:
            return 0

        return sum(self.hungryHistory) / len(self.hungryHistory)
    
    def calculateTimeToLive(self) -> float:
        if self.timeDied is None:
            return 0
        
        return self.timeDied - self.timeBorn
    
    def _eat(self):
        # Increment this snake's total number of food eaten by 1
        self.numberOfFoodEaten += 1
        # Record number of hungry moves
        self.hungryHistory.append(self.hungryMovesTaken)
        # Reset this snake's number of hungry moves to 0
        self.hungryMovesTaken = 0
        # Min distance becomes 0
        self.distances.append(0)
        # Reset positionHistory
        self.positionHistory = []
        # Reset this snake's starting position to its current position
        self.startPosition = (self.x, self.y)

    # Perform game logic from one step
    def update(self):
        # Verify that there is another available active Food object at this snake's food level
        # Keep appending another Food object to the game's list of active food sources until there is another available active Food object at this snake's food level
        while len(Snake._activeFood) <= self.getNumberOfFoodEaten():
            Snake._activeFood.append(Food._spawn())

        # Update this snake's target food
        self.targetFood = Snake._activeFood[self.getNumberOfFoodEaten()]

        # Check for food collision
        if self == self.targetFood:
            self._eat()

        # Update whether or not this snake has died
        if self._isDead():
            self.dead = True
            self.timeDied = datetime.timestamp(datetime.now())
            self.fitness = self._calculateFitness()
            return

        # Increment this snake's number of moves and moves w/o food by 1
        self.totalMovesTaken += 1
        self.hungryMovesTaken += 1

        # Make a decision
        self._think(self.targetFood)

        # Enable autopilot if this snake is caught in a loop or if it has eaten food
        self.autopilot = self._isCaughtInLoop(threshold=3) or self.getNumberOfFoodEaten() != 0

        # Make a move based on this snake's current direction
        move_set = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.x += move_set[int(self.direction)][0]
        self.y += move_set[int(self.direction)][1]

        # Update the closest this snake has been to it's target food
        self.distances.append(self._getDistanceToFood(self.targetFood))
        self.positionHistory.append((self.x, self.y))

        # Update the snake's body coordinates
        self.bodyCoordinates.append((self.x, self.y))
        while len(self.bodyCoordinates) > self.numberOfFoodEaten + 1:
            self.bodyCoordinates.pop(0)

    # Returns whether or not the snake is dead
    def _isDead(self) -> bool:
        isSnakeTouchingItself = (self.x, self.y) in self.bodyCoordinates[:-1] # Die if touching self
        isSnakeStarving = self.hungryMovesTaken > 10000000 # Die if too many moves w/o food
        isSnakeOutOfBounds = not Snake._inBounds(self.x, self.y) # Die if snake goes out of game bounds
        
        return isSnakeTouchingItself or isSnakeStarving or isSnakeOutOfBounds
    
    # Draws visual representation of this Snake object to the running pygame window
    def _draw(self):
        # Draw the body
        count = 0
        for x, y in self.bodyCoordinates:
            count += 1
            try:
                # Draw rect to screen
                if self.getNumberOfFoodEaten() + 1 >= len(Snake._activeFood):
                    pygame.draw.rect(screen, (self.neuralNetworkObject.id.r, self.neuralNetworkObject.id.g, self.neuralNetworkObject.id.b), (x * SCALE, y * SCALE, SCALE, SCALE))
                else:
                    pygame.draw.rect(screen, (20, 20, 20), (x * SCALE, y * SCALE, SCALE, SCALE))
            except ValueError:
                pass
            
        # Draw the head of the snake
        try:
            if self.getNumberOfFoodEaten() + 1 >= len(Snake._activeFood):
                pygame.draw.rect(screen, (self.neuralNetworkObject.id.r, 255, self.neuralNetworkObject.id.b), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
            else:
                pygame.draw.rect(screen, (60, 60, 60), (self.x * SCALE, self.y * SCALE, SCALE, SCALE))
        except ValueError:
            pass

    def _isCaughtInLoop(self, threshold) -> bool:
        return self.positionHistory.count((self.x, self.y)) > threshold
    
    # String representation
    def __str__(self):
        return f"Snake(X: {self.x} Y: {self.y})\nFitness: {self.fitness}\nIs Dead?: {self.dead}\nNumber of Food Eaten: {self.numberOfFoodEaten}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()

    # Returns whether or not the given 2D coordinates lie within the game space
    @staticmethod
    def _inBounds(x: int, y: int) -> bool:
        return x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT

