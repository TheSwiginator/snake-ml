import numpy as np, json, sys, time, pygame, os, glob
from .constants import *
from .population import SLPopulation
from .snake_obj import Snake
from .food_obj import Food
from .utils import screen, clock
from datetime import datetime

# The SLPopulation is an that computes and stores snake analytics for a current simulation
population = SLPopulation(100, 12, 16, 4, 0.5) # Constructs a new population object
filename = None # File name of the desired loaded json file, containing the snake populations' neural net info and generation info

# Check if the user chooses to load the most recent json data
if len(sys.argv) == 1:
    # Get list of json files in the sl_data directory
    files = glob.glob('./snake-ml/sl_data/*.json')
    # Verify that there is a save in the directory to load from
    if len(files) != 0:
        # Sort the files by created date and parse the most recent file string for the filename
        filename = max(files, key=os.path.getctime).split("\\")[1]
        print(f"No filename parameter was specified. Loading from most recent population from {filename}")

# Check if the user wishes to load a specific json file
if len(sys.argv) == 2:
    # Accept the file path as a command-line input
    filename = sys.argv[1]
    print(f"File parameter specified. Loading from {filename}")

# Check if the user neither has an existing savepoint nor provided a specifc file path
if filename is None:
    # Warn the user
    print("Warning: No savepoints have been located in the current directory. Population will start fresh.")
else:
    # Load the json data in the current SLPopulation object
    population.loadJSON(filename)
    
time.sleep(5)

# Main loop
run = True
while run:
    try:
        # handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        # Check if all the snakes are dead
        if population.areAllSnakesDead():
            # Perform natural selection on the current population of snakes
            population.naturalSelection()
            # Reset number of Food objects to 0
            Food._count = 0
            # Most recent Food object created gets get carried over to new list of active Food objects
            Snake._activeFood = [Food(Snake._activeFood[-1].x, Snake._activeFood[-1].y)]
        
        # Update all the live snakes in the simulation
        population.updatePopulation()
        # Fill screen with black
        screen.fill((0, 0, 0))
        # Draw draw the all the Game Objects
        population.draw()
        # Print Population Stats
        os.system("cls")
        print(population)
        # Update the screen
        pygame.display.update()
        # Set the frame rate
        clock.tick(60)
    except KeyboardInterrupt:
        # Pressing Ctrl+C will cause the simulation to autosave then exit the program
        run = False

# Generic file naming scheme
PATH = f"population_{population.generation}.json"
print(f"Saving population to {PATH}...")
# Save population data to new json file
if population.saveJSON(PATH) != None:
    print("Save complete!")
else:
    print("ERROR: Save failed!")
    time.sleep(10)