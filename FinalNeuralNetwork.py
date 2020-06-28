import numpy as np
from enum import Enum
import math
import copy
import random

class Activation(Enum):
    SIGMOID = 1
    RELU = 2
    TANH = 3
    SOFTMAX = 4
    NONE = 5
class Neuron:
    neuronID = 0
    def __init__(self, numInputValues, activationFunction, noWeights=False):
        if not noWeights:
            
            self.weights = np.random.randn(numInputValues, 1) * np.sqrt(2 / (numInputValues))
        else:
            self.weights = np.ones((numInputValues, 1))
        self.bias = 1
        
        self.activationFunction = activationFunction
        self.numInputValues = numInputValues
        self.dataPoints = []
        self.id = Neuron.neuronID
        Neuron.neuronID += 1

    def outputPrediction(self):
        if len(self.dataPoints) < len(self.weights):
            return "Not enough dataPoints for neuron to make a prediction"
        tempDataPoints = self.dataPoints
        preFunctionValue = np.dot(np.array(self.dataPoints), np.array(self.weights)) + self.bias
        self.clearDataPoints()
        if self.activationFunction is Activation.SIGMOID:
            return [tempDataPoints, self.weights, 1/(1 + np.exp(-preFunctionValue))]
        if self.activationFunction is Activation.RELU:
            return [tempDataPoints, self.weights, max(0.02 * preFunctionValue, preFunctionValue)]
        if self.activationFunction is Activation.TANH:
            return [tempDataPoints, self.weights, np.tanh(preFunctionValue)]
        if self.activationFunction is Activation.NONE or self.activationFunction is Activation.SOFTMAX:
            return [tempDataPoints, self.weights, preFunctionValue]

    def setWeights(self, weights):
        self.weights = weights
    
    def getWeights(self):
        return self.weights

    def setBias(self, bias):
        self.bias = bias
    
    def getBias(self):
        return self.bias
    
    def addDataPoint(self, data):
        if len(self.dataPoints) < self.numInputValues:
            self.dataPoints.append(data)
    
    def clearDataPoints(self):
        self.dataPoints = []
    def getID(self):
        return self.id
    
    def getActivation(self):
        return self.activationFunction
    
    def __str__(self):
        return str(self.id)

class NeuralNetwork:
    netDictionary = dict()
    predDictionary = dict()
    def __init__(self, numInputValues, learningRate):
        self.numInputValues = numInputValues
        self.layers = []
        self.learningRate = learningRate
        self.addLayer(numNeurons = self.numInputValues, numInputValues = 1, activationFunction = Activation.NONE, firstLayer = True)
    
    def addLayer(self, numNeurons, numInputValues, activationFunction, firstLayer):
        layer = []
        if numInputValues == 0:
            numInputValues = len(self.layers[len(self.layers) - 1])
        for i in range(0, numNeurons):
            if firstLayer:
                neuron = Neuron(1, Activation.NONE, noWeights = True)
                NeuralNetwork.netDictionary[neuron.getID()] = [len(self.layers), i]
                layer.append(neuron)
            else:
                neuron = Neuron(numInputValues, activationFunction)
                NeuralNetwork.netDictionary[neuron.getID()] = [len(self.layers), i]
                layer.append(neuron)
        self.layers.append(layer)
    def applySoftmax(self, X):
        return (np.exp(X) / np.sum(np.exp(X)))
    def predict(self, X):
        if len(X) < len(self.layers[0]):
            return "Number of values in X doesn't match # of neurons in first layer of neural network"
        
        for i in range(0, len(self.layers[0])):
            self.layers[0][i].addDataPoint(X[i])

        predictions = []
        for i in range(1, len(self.layers)):
            lastLayerPredictions = []
            for neuron in self.layers[i - 1]:
                prediction = neuron.outputPrediction()
                lastLayerPredictions.append(prediction[2])
                NeuralNetwork.predDictionary[neuron.getID()] = prediction
            for neuron in self.layers[i]:
                for pred in lastLayerPredictions:
                    neuron.addDataPoint(float(pred))
        predictions = []
        for neuron in self.layers[len(self.layers) - 1]:
            neuronPrediction = neuron.outputPrediction()
            predictions.append(neuronPrediction[2])
            NeuralNetwork.predDictionary[neuron.getID()] = neuronPrediction
        if self.layers[len(self.layers) - 1][0].getActivation() is Activation.SOFTMAX:
            predictions = self.applySoftmax(np.array(predictions))
            for i in range(0, len(self.layers[len(self.layers) - 1])):
                NeuralNetwork.predDictionary[self.layers[len(self.layers) - 1][i].getID()][2] = predictions[i]
        return np.array(predictions).reshape(len(self.layers[len(self.layers) - 1]), 1)
    def fit(self, X, y):
        if len(X) != len(y):
            print("Input and output lengths don't match")
            return "Input and output lengths don't match"
        print(f"Before entering for loop")
        for i in range(0, len(X)):
            X_train = X[i]
            print(f"X_train is {X[i]}")
            y_train = y[i]
            print(f"Y_train is {y[i]}")
            pred_value = self.predict(X_train)
            errors = self.calculateError(y_train, pred_value)  
            total_error = sum(errors)
            self.calculateDifferentials(errors, y_train, pred_value)
            for i in range(1, len(self.layers)):
                layer = self.layers[i]
                for neuron in layer:
                    currentWeights = neuron.getWeights()
                    weightDifferentials = NeuralNetwork.netDictionary[neuron.getID()][2]
                    print(f"Weight differentials are {weightDifferentials} for neuron with id {neuron.getID()}")
                    newWeights = np.array(currentWeights) - self.learningRate * np.array(weightDifferentials)
                    
                    neuron.setWeights(newWeights)
                    
    def calculateError(self, targetY, actualY):
        targetY = np.array(targetY)
        actualY = np.array(actualY)
        if (self.layers[len(self.layers) - 1][0].getActivation() is Activation.SOFTMAX):
            print("Softmax Activation")
        error = 0.5 * (np.square(targetY.flatten() - actualY.flatten()))
        return error
    def calculateLastLayerWeightDifferentials(self, errors, targetY, actualY):
        differentials = list()
        for i in range(0, len(self.layers[len(self.layers) - 1])):
            neuron = self.layers[len(self.layers) - 1][i]
            weights = neuron.getWeights()
            neuronPrediction = NeuralNetwork.predDictionary[neuron.getID()]
            errorToUse = errors[i]
            errorOutputDifferential = -(targetY[i] - actualY[i])
            outputInputDifferential = neuronPrediction[2] * (1 - neuronPrediction[2])
            weightDifferentials = []
            for i in range(0, len(weights)):
                inputWeightDifferential = neuronPrediction[0][i]
                totalDifferential = errorOutputDifferential * outputInputDifferential * inputWeightDifferential
                differentials.append(totalDifferential)
                weightDifferentials.append(totalDifferential)
            NeuralNetwork.netDictionary[neuron.getID()].append(weightDifferentials)
        return differentials
    def calculateDifferentials(self, errors, targetY, actualY):
        differentials = []
        lastlayerDifferentials = self.calculateLastLayerWeightDifferentials(errors, targetY, actualY)
        for i in range(1, len(self.layers)):
            if len(self.layers) - 1 - i == 0:
                continue
            layer = self.layers[len(self.layers) - 1 - i]
            for j in range(0, len(layer)):
                neuron = layer[j]
                weights = neuron.getWeights()
                neuronPrediction = NeuralNetwork.predDictionary[neuron.getID()]
                errorOutputDifferential = 0
                allPaths = []
                for l in range(0, len(self.layers[len(self.layers) - 1])):
                    paths = self.tracePath(len(self.layers) - 1 - i, j, len(self.layers) - 1, l)
                    for path in paths:
                        allPaths.append(path)
                for path in allPaths:
                    lastLayerNeuronPosition = NeuralNetwork.netDictionary[path[0].getID()][1]
                    pathDifferential = self.calculateErrorOutputDifferentialAlongPath(path, targetY, actualY, lastLayerNeuronPosition)
                    errorOutputDifferential += pathDifferential
                if neuron.getActivation() is Activation.SIGMOID:
                    outputInputDifferential = neuronPrediction[2] * (1 - neuronPrediction[2])
                elif neuron.getActivation() is Activation.RELU:
                    if neuronPrediction[2] > 0:
                        outputInputDifferential = 1
                    else:
                        outputInputDifferential = 0.02
                else:
                    outputInputDifferential = 0
                weightDifferentials = []
                for k in range(0, len(weights)):
                    inputWeightDifferential = neuronPrediction[0][k]
                    totalDifferential = errorOutputDifferential * outputInputDifferential * inputWeightDifferential
                    weightDifferentials.append(totalDifferential)
                NeuralNetwork.netDictionary[neuron.getID()].append(weightDifferentials)
    def calculateErrorOutputDifferentialAlongPath(self, path, targetY, actualY, lastLayerPosition):
        edgeNeuron = self.layers[len(self.layers) - 1][lastLayerPosition]
        errorOutputDifferential = -(targetY[lastLayerPosition] - actualY[lastLayerPosition])
        differential = errorOutputDifferential
        for i in range(0, len(path) - 1):
            neuron = path[i]
            neuronPrediction = NeuralNetwork.predDictionary[neuron.getID()]
            outputInputDifferential = neuronPrediction[2] * (1 - neuronPrediction[2])
            nextNeuronLayerPosition = NeuralNetwork.netDictionary[path[i + 1].getID()][1]# Create functionality to determine neuron positioning based on neuron id
            inputOutputDifferential = neuron.getWeights()[nextNeuronLayerPosition]
            differential *= outputInputDifferential * inputOutputDifferential
        
        return differential
    def getLayers(self):
        return self.layers
    def tracePath(self, layerNeuron1, positionNeuron1, layerNeuron2, positionNeuron2):
        if layerNeuron1 == layerNeuron2 and positionNeuron1 == positionNeuron2:
            return list()
        neuron1ID = self.layers[layerNeuron1][positionNeuron1].getID()
        currentPath = list()
        allPaths = list()
        self.tracePathHelper(layerNeuron1, positionNeuron1, layerNeuron2, positionNeuron2, currentPath, allPaths)
        paths = []
        helper = []
        for i in range(0, len(allPaths[0])):
            paths.append(allPaths[0][i])
            if allPaths[0][i].getID() == neuron1ID:
                helper.append(paths)
                paths = []
        truePaths = []
        referencePath = helper[0]
        truePaths.append(referencePath)
        changeReference = False
        for i in range(1, len(helper)):
            path = helper[i]
            if (len(path) > len(helper[i - 1])):
                changeReference = True
            else:
                changeReference = False
            copyRefPath = copy.deepcopy(referencePath)
            copyRefPath[-len(path):] = path
            truePaths.append(copyRefPath)
            if changeReference: 
                referencePath = copyRefPath
        return truePaths
    def tracePathHelper(self, layerNeuron1, positionNeuron1, layerNeuron2, positionNeuron2, currentPath, allPaths):
        if layerNeuron1 == layerNeuron2 and positionNeuron1 == positionNeuron2:
            currentPath.append(self.layers[layerNeuron2][positionNeuron2])
            allPaths.append(currentPath)
            return
        if (layerNeuron1 == layerNeuron2 and positionNeuron1 != positionNeuron2) or (layerNeuron2 < layerNeuron1):
            return
        neuron1 = self.layers[layerNeuron1][positionNeuron1]
        neuron2 = self.layers[layerNeuron2][positionNeuron2]
        currentPath.append(neuron2)
        for i in range(0, len(self.layers[layerNeuron2 - 1])):
            self.tracePathHelper(layerNeuron1, positionNeuron1, layerNeuron2 - 1, i, currentPath, allPaths)

