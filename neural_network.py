import numpy as np
import random
from .utils import RGB

class NeuralNetwork:
    _count = 0 # Total number of SingleLayerNeuralNetwork instances

    def __init__(self, inputNodes, hiddenNodes, outputNodes, mutationRate, mother=RGB(-1,-1,-1), father=RGB(-1,-1,-1)):
        self.inputNodes = inputNodes # Number of nodes in input layer
        self.hiddenNodes = hiddenNodes # Number of nodes in hidden layer
        self.outputNodes = outputNodes # Number of nodes in output layer
        self.mutationRate = mutationRate # Mutation rate of this SingleLayerNeuralNetwork
        self.mother = mother # SingleLayerNeuralNetwork id of mother
        self.father = father # SingleLayerNeuralNetwork id of father
        self.id = RGB(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if mother.r == -1 else random.choice([mother, father]) # SingleLayerNeuralNetwork id of this SingleLayerNeuralNetwork
        self.weightsInputToHidden = np.random.rand(self.hiddenNodes, self.inputNodes) # Weights from input layer to hidden layer
        self.weightsHiddenToOutput = np.random.rand(self.outputNodes, self.hiddenNodes) # Weights from hidden layer to output layer
        self.biasHidden = np.random.rand(self.hiddenNodes, 1) # Biases of hidden layer
        self.biasOutput = np.random.rand(self.outputNodes, 1) # Biases of output layer
        NeuralNetwork._count += 1 # Increment total number of SingleLayerNeuralNetwork objects by 1

    # Perform a feedforward operation on this SingleLayerNeuralNetwork object and return the output
    def feedForward(self, inputArray):
        # Convert input array to numpy array and transpose it
        inputFeed = np.array(inputArray, ndmin=2).T

        # Feed input layer to hidden layer
        hiddenFeed = np.dot(self.weightsInputToHidden, inputFeed)
        hiddenFeed = np.add(hiddenFeed, self.biasHidden)
        hiddenFeed = self._sigmoid(hiddenFeed)

        # Feed hidden layer to output layer
        outputFeed = np.dot(self.weightsHiddenToOutput, hiddenFeed)
        outputFeed = np.add(outputFeed, self.biasOutput)
        outputFeed = self._sigmoid(outputFeed)

        return outputFeed
    
    # Determines the activation of a node
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Vectorized function to mutate a value
    def _mutate(self, val):
        # Mutate value if random number is less than mutation rate
        if random.random() < self.mutationRate:
            color_swap = True
            return val + random.gauss(0, 0.1)
        return val

    # Potentially mutate the weights and biases of this SingleLayerNeuralNetwork object
    def mutate(self):
        colorSwap = False
        # Vectorize the mutate function
        mutate = np.vectorize(self._mutate)

        if colorSwap:
            self.id.set_value(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Set weights and biases to new mutated numpy array
        self.weightsInputToHidden = mutate(self.weightsInputToHidden)
        self.weightsHiddenToOutput = mutate(self.weightsHiddenToOutput)
        self.biasHidden = mutate(self.biasHidden)
        self.biasOutput = mutate(self.biasOutput)


    # Returns a copy of this SingleLayerNeuralNetwork object
    def clone(self):
        # Create new SingleLayerNeuralNetwork object with identical instance variables
        mirroredNeuralNetwork = NeuralNetwork(self.inputNodes, self.hiddenNodes, self.outputNodes, self.mutationRate)
        # Copy the weights and biases
        mirroredNeuralNetwork.weightsInputToHidden = np.copy(self.weightsInputToHidden)
        mirroredNeuralNetwork.weightsHiddenToOutput = np.copy(self.weightsHiddenToOutput)
        mirroredNeuralNetwork.biasHidden = np.copy(self.biasHidden)
        mirroredNeuralNetwork.biasOutput = np.copy(self.biasOutput)
        mirroredNeuralNetwork.mother, mirroredNeuralNetwork.father = self.id, self.id
        return mirroredNeuralNetwork
    
    # String representation
    def __str__(self):
        return f"weightsInputToHidden: {self.weightsInputToHidden}, weightsHiddenToOutput: {self.weightsHiddenToOutput}, biasHidden: {self.biasHidden}, biasOutput: {self.biasOutput}"
    
    # Printable representation
    def __repr__(self):
        return self.__str__()
    
    # Dictionary representation
    def __dict__(self):
        # This function enables models to be saved as JSON files
        return {
            "input_nodes": self.inputNodes,
            "hidden_nodes": self.hiddenNodes,
            "output_nodes": self.outputNodes,
            "weights_ih": self.weightsInputToHidden.tolist(),
            "weights_ho": self.weightsHiddenToOutput.tolist(),
            "bias_h": self.biasHidden.tolist(),
            "bias_o": self.biasOutput.tolist()
        }