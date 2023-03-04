from .neural_network import NeuralNetwork
from .snake_obj import Snake
from .constants import *
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pygame

class SLPopulation:
    def __init__(self, size: int, inputNodes, hiddenNodes, outputNodes, mutation_rate: float, population=[]):
        self.size = size # Number of snakes in this population
        self.inputNodes = inputNodes # Number of nodes in input layer
        self.hiddenNodes = hiddenNodes # Number of nodes in hidden layer
        self.outputNodes = outputNodes # Number of nodes in output layer
        self.mutation_rate = mutation_rate # Mutation rate of this population
        self.generation = 0 # Current generation of this population
        self.best = None # Best snake in this population
        self.bests = [] # Initialized to single snake
        self.population = population # Set population list to imported population
        # Set population list to None if we're not importing a JSON fileself.dead
        if len(self.population) == 0:
            self.population = [Snake(nn=NeuralNetwork(inputNodes, hiddenNodes, outputNodes, mutation_rate), generation=self.generation) for i in range(size)]
    
    # Loads a population from a json file
    def loadJSON(self, filename: str):
        data = None
        # Checks if filename doesn't exist
        if not os.path.exists(f"./snake-ml/sl_data/{filename}"):
            return data
        
        # Load json data from file and saves it to data variable
        with open(f"./snake-ml/sl_data/{filename}", "r") as f:
            data = json.load(f)

        # Set population attributes
        self.generation = data["generation"]
        for i, v in enumerate(data["population"]):
            self.population[i].neuralNetworkObject.inputNodes = v['input_nodes']
            self.population[i].neuralNetworkObject.hiddenNodes = v['hidden_nodes']
            self.population[i].neuralNetworkObject.outputNodes = v['output_nodes']
            self.population[i].neuralNetworkObject.weightsInputToHidden = np.array(v['weights_ih'])
            self.population[i].neuralNetworkObject.weightsHiddenToOutput = np.array(v['weights_ho'])
            self.population[i].neuralNetworkObject.biasHidden = np.array(v['bias_h'])
            self.population[i].neuralNetworkObject.biasOutput = np.array(v['bias_o'])

        return self
    
    # Saves a population to a json file
    def saveJSON(self, filename: str):
        # Save population.population to json
        with open(f"./snake-ml/sl_data/{filename}", "w") as f:
            json.dump(self.__dict__(), f)
            return True
    
    # Determine the genetic composition of the descendants of the current generation
    def naturalSelection(self) -> None:
        # Corner case: if there is no population, return 0
        if len(self.population) == 0: return
        # Holds the child snakes
        futureGeneration = []
        # Fittest snake survives and is recorded
        self.best = self._generationMostFit() # Set the generations' best snake
        self.bests.append(self.best) # Add best to list of best snakes
        # Keep list of best snakes within a certain size
        if len(self.bests) > 2000:
            self.bests.pop(0)
        # Add children to future generation
        enable_autopilot = self.bests[-1].fitness == self.bests[-2].fitness if len(self.bests) > 1 else False # Enable autopilot if the best snake hasn't improved in the 2 most recent generations
        for i in range(self.size):
            parent_a: Snake = self._selectParent() # Choose first snake from population
            parent_b: Snake = self._selectParent() # Choose second snake from population
            child: Snake = self._breed(parent_a, parent_b) # Mix genes of two parents
            child.autopilot = enable_autopilot # Enable autopilot if the best snake hasn't improved in the 2 most recent generations
            child.neuralNetworkObject.mutate() # Mutate child snake
            futureGeneration.append(child) # Add child
        
        # Set the current generation to the new list of child snakes
        self.population = futureGeneration
        # Preserve the best snake for the next generation
        
        self.population[0] = Snake(nn=self.best.neuralNetworkObject, generation=(self.generation + 1))
        # Increment generation count by 1
        self.generation += 1

    # Returns if all snakes in the population are dead
    def areAllSnakesDead(self):
        return all([snake.dead for snake in self.population])
    
    # Updates the states of all the Snake objects in the population
    def updatePopulation(self) -> None:
        for snake in self.population:
            if not snake.dead:
                snake.update()

    # Draw all the Game Objects to the pygame display
    def draw(self) -> None:
        # Draw all the snakes in the population that are not dead
        for snake in self.population:
            if not snake.dead:
                snake._draw()

        # Draw all the food that are active
        self._drawActiveFood()

    # Mix the genes of two snakes to create a new snake
    def _breed(self, parent_a: Snake, parent_b: Snake) -> Snake:
        # Create new NeuralNetwork object
        child_nn = NeuralNetwork(self.inputNodes, self.hiddenNodes, self.outputNodes, self.mutation_rate, parent_a.neuralNetworkObject.id, parent_b.neuralNetworkObject.id)
        child = Snake(nn=child_nn, generation=(self.generation + 1))

        # Child inherits average weights and bias of the parents
        child.neuralNetworkObject.weightsInputToHidden = np.add(parent_a.neuralNetworkObject.weightsInputToHidden, parent_b.neuralNetworkObject.weightsInputToHidden) / 2
        child.neuralNetworkObject.weightsHiddenToOutput = np.add(parent_a.neuralNetworkObject.weightsHiddenToOutput, parent_b.neuralNetworkObject.weightsHiddenToOutput) / 2
        child.neuralNetworkObject.biasHidden = np.add(parent_a.neuralNetworkObject.biasHidden, parent_b.neuralNetworkObject.biasHidden) / 2
        child.neuralNetworkObject.biasOutput = np.add(parent_a.neuralNetworkObject.biasOutput, parent_b.neuralNetworkObject.biasOutput) / 2

        return child

    # Selects a parent weighted by fitness
    def _selectParent(self) -> Snake:
        # Corner case: if there is no snakes yet loaded into the population, return a random snake
        if self.best is None: return random.choice(self.population)
        # Use map function to map self.population to a list of tuples containing the fitness deviation and its corresponding snake
        fitnessDeviations = list(map(lambda x: (x.fitness - self._getAverageBestFitness(), x), self.population))
        # Then use the filter function to filter out all the snakes with a fitness deviation lower than 0
        positiveFitnessDeviations = list(filter(lambda x: x[0] > 0, fitnessDeviations))
        # Then use the map function to map the filtered list to a list of tuples containing a random integer from 0 to the fitness deviation and its corresponding snake
        randomDeviations = list(map(lambda x: (random.randint(0, int(x[0])**2), x[1]), positiveFitnessDeviations))
        # Then use the max function to return the snake with the highest random integer
        if len(randomDeviations) > 0:
            return max(randomDeviations, key=lambda x: x[0])[1]
        
        # If there are no positive deviations, then return a random snake
        return self.best

    # Draws all the active food to the pygame display
    def _drawActiveFood(self):
        for food in Snake._activeFood:
            food.draw()

    # Returns the best snake object in the last 2000 generations
    def _getGoat(self):
        return max(self.bests, key=lambda x: x.fitness)
    
    # Returns the average fitness of this population
    def _getAverageBestFitness(self) -> float:
        # Corner case: if there are no best snakes, return 0
        if len(self.bests) == 0: return 0
        return sum([snake.fitness for snake in self.bests]) / len(self.bests)
    
    # Returns the highest fitness of all generations
    def _overallMostFitness(self):
        # Corner case: if there are no best snakes, return 0
        if len(self.bests) == 0: return 0
        return max([snake.fitness for snake in self.bests])
    
    # Returns the most number of food eaten by a snake
    def _overallMostFoodEaten(self):
        # Corner case: if there are no best snakes, return 0
        if len(self.bests) == 0: return 0
        return max([snake.getNumberOfFoodEaten() for snake in self.bests])
    
    # Returns the average max number of food eaten per generation
    def _overallAverageFoodEaten(self):
        # Corner case: if there are no best snakes, return 0
        if len(self.bests) == 0: return 0
        return sum([snake.getNumberOfFoodEaten() for snake in self.bests]) / len(self.bests)
    
    # Returns the average max fitness per generation
    def _overallAverageFitness(self):
        # Corner case: if there are no best snakes, return 0
        if len(self.bests) == 0: return 0
        return sum([snake.fitness for snake in self.bests]) / len(self.bests)
    
    # Returns the average time alive for a generations' most fit snake
    def _overallAverageSnakeLifespan(self):
        # Corner case: if there are no best snakes, return 0
        if len(self.bests) == 0: return 0
        return sum([snake.calculateTimeToLive() for snake in self.bests]) / len(self.bests)

    # Returns the average time without eating for a generations' most fit snake
    def _overallAverageTimeToFood(self):
        # Corner case: if there are no best snakes, return 0
        if len(self.bests) == 0: return 0
        return sum([snake.calculateTimeToFood() for snake in self.bests]) / len(self.bests)
    
    # Returns the fittest snake in this population
    def _generationMostFit(self) -> Snake:
        return max(self.population, key=lambda x: x.fitness)
        
    # String representation
    def __str__(self):
        enable_autopilot = self.bests[-1].fitness == self.bests[-2].fitness if len(self.bests) > 1 else False
        return f"Auto Pilot Enabled?: {enable_autopilot}\nGeneration: {self.generation}\nAverage Fitness: {self._getAverageBestFitness()}\nOverall Best Fitness: {self._overallMostFitness()}\nOverall Most Food Eaten: {self._overallMostFoodEaten()}\nAverage Number of Food Eaten: {self._overallAverageFoodEaten()}\nAverage Fitness: {self._getAverageBestFitness()}\nAverage Snake Time Alive: {self._overallAverageSnakeLifespan()}\nAverage Time to Food: {self._overallAverageTimeToFood()}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "generation": self.generation,
            "population": [snake.neuralNetworkObject.__dict__() for snake in self.population],
        }