#create genotype object
#create a population - random init
#write fitness function - will use knapsack problem, so fitness function will be based of value(MVP1)
#write parent selection func - try a few, start with the roullette wheel selection
#write reproduce/crossover function - will start with one point crossover, then advance
#write mutation function - bit flip method
#use matplotlib to display the results

import random
import matplotlib
import numpy as np
import math
from numpy.lib.function_base import average

#create an item which will store the value and weight parameters, is to be added to bag
class object: 
    def __init__(self, value, weight, objID):
        self.value = value
        self.weight = weight

#create a bag item, will be filled with objects. Solutions will be oprimized for it
class Bag(): 
    def __init__(self, size):
        self.size = size
        self.capacity = size
        self.value = 0
        self.items = []

    #add to bag, reduce bag capacity by the weight of the item
    def addToBag(self, item): 
        self.items.append(item)
        self.value = self.value + item.value
        self.capacity = self.capacity - item.weight

    def isFull(self, item):
        if self.capacity - item.weight <= 0:
            return True
        else:
            return False

#create sample objects, will use a different method once done
try0 = object(4, 5, 0)
try1 = object(9, 2, 1)
try2 = object(3, 8, 2)
try3 = object(2, 3, 3)
try4 = object(5, 4, 4)
try5 = object(1, 8, 5)
try6 = object(7, 3, 6)
try7 = object(4, 9, 7)
try8 = object(2, 2, 8)

items = [try0, try1, try2, try3, try4, try5, try6, try7, try8]

#bubble sort to sort objects by value
def sortItems(items):
    n = len(items)

    for i in range(n - 1):
        for j in range(n - i - 1):
            if items[j].value > items[j + 1].value:
                temp = items[j]
                items[j + 1] = items[j]
                items[j] = temp

    return items

#fill the bag with created items
def fillBag(size, items):
    newBag = Bag(size)

    sortedItems = sortItems(items)

    for i in sortedItems:
        if newBag.isFull == False:
            newBag.addToBag(i)

#get the fitness for a genotype
def fitness(valueAndGene):
    popFitnesses = []

    for i in range(len(valueAndGene)):
        genotype = valueAndGene[i]

        value = genotype[0]
        weight = 8#temp value before i add the weight to the array
        
        genotypeFitness = value - weight #largest value has the greatest fitness
        popFitnesses.append(genotypeFitness)

    return popFitnesses 

# def fitness(bag):
#     fitness = 0
#     return fitness

# def addFitness(valueAndGene):
#     fitnesses = fitness(valueAndGene)
    
#     for i in range(len(valueAndGene)):
#         certainValueAndGene = valueAndGene[i] #cant add value to tuple for some reason, throws an error
#         certainFitness = fitnesses[i]
#         certainValueAndGene = certainValueAndGene + certainFitness

#     return valueAndGene  

#creating a poulation of solutions to the problem, used random populating
def createPopulation(popSize, geneNumber):
    population = []

    for i in range(popSize):
        genotype = []

        for j in range(geneNumber):
            gene = random.randint(0, 1)
            genotype.append(gene)
        
        population.append(genotype)

    return population

#filling the bag from genotype to be able to get the fitness for a solution
def fillFromGenotype(bag, genotype, items):
    for i in range(len(genotype)):
        gene = genotype[i]

        item = items[i]

        if gene == 1 and bag.isFull(item) == False:
            bag.addToBag(item)

#filling the whole population with the generated genotypes, size is declared as one of the parameters
def fillForPopulation(population, size):
    valueAndGene = []

    for genotype in population:
        bag = Bag(size)

        fillFromGenotype(bag, genotype, items)

        valueAndGene.append([bag.value, genotype, bag.value - 32])

    return valueAndGene

#bubble sort to sort genotypes by descending fitness    
def sortPairs(valueAndGene):
    n = len(valueAndGene)

    for i in range(n - 1):
        for j in range(n - i - 1):
            fitness1 = valueAndGene[j]
            fitness2 = valueAndGene[j + 1]
            if fitness1[2] > fitness2[2]: #revise what the array looks like and see what index i need
                temp = valueAndGene[j]
                valueAndGene[j + 1] = valueAndGene[j]
                valueAndGene[j] = temp
    
    return valueAndGene

#function to get elites, percentage of elites is declared as a parameter. Elites will be passed onto next gen
def getElites(valueAndGene, elitismP):
    sortedFitnesses = sortPairs(valueAndGene)

    numOfElite = elitismP * len(valueAndGene)

    eliteValueAndGene = []
    
    for i in range(int(numOfElite)):
        eliteFitness = sortedFitnesses[i]
        eliteValueAndGene.append(eliteFitness)

    return eliteValueAndGene

#function to clone the population, used to append children to population for a new cycle
def clone(population, newPopSize):
    numOfOffsprings = int(newPopSize/population)

    newPopulation = []

    for i in range(len(population)):
        genotype = population[i]
        for j in range(numOfOffsprings):
            offspring = genotype.copy()
            newPopulation.append(offspring)

    return newPopulation

#crossover function to create children
def breed(population, newPopSize, valueAndGene): #
    newPopulation = []

    sortedFitness = sortPairs(valueAndGene) #sort values, genes by fitness descending

    totalFitness = 0

    probabilities = []

    for i in range(len(population)): #calculate total fitness
        genotype = sortedFitness[i]
        totalFitness += genotype[2]

    for j in range(len(population)): #calculate the probability for each genotype to get selected, roulette wheel method
        fitness = valueAndGene[j]
        probabilities.append(fitness[2]/totalFitness)

    for x in range(newPopSize): #add half of the mother genotype and half of father genotype to child
        father = random.choices(population, weights=probabilities)
        mother = random.choices(population, weights=probabilities)

        child = []

        index = len(father)/2

        for k in range(0, int(index)):
            child.append(father[k])

        for l in range(int(index), len(mother)):
            child.append(mother[l])

        newPopulation.append(child)

    return newPopulation

def mutate(population, mutationP): #function to mutate a random bit in a genotype, used to maintain diverity. mutation rate is a declared parameter in main loop
    
    numOfMutate = mutationP * len(population)

    for i in range(math.floor(numOfMutate)):
        popIndex = random.randint(0, len(population) - 1)
        genotype = population[popIndex]

        bit = random.randint(0, len(genotype))

        genotype[bit] = not genotype[bit] 

    return population

def getFitnessStats(population, items, size):
    mostFitGenotype = []
    bestFitness = 0
    totalPopFitness = 0

    for genotype in population:
        bag = Bag(size)

        fillFromGenotype(bag, genotype, items)

        if bag.value > bestFitness:
            bestFitness = bag.value
            mostFitGenotype = genotype

        totalPopFitness = totalPopFitness + bag.value

        averageFitness = totalPopFitness/len(population)

    return (bestFitness, mostFitGenotype, averageFitness)

def fillByGenetics(size, items):
    popSize = 100
    numOfGenes = len(items) 
    maxGenerations = 100
    convergenceThreshold = 10
    elitismP = 0.1
    mutateP = 0.05

    bestFitness = 0
    mostFitGenotype = []
    bestAverageFitnessForPop = 0
    nonImprovingGens = 0
    bag = Bag(size)

    population = createPopulation(popSize, numOfGenes)

    for i in range(maxGenerations):
        valueAndGene = fillForPopulation(population, size)

        population = getElites(valueAndGene, elitismP)

        population = breed(population, popSize, valueAndGene)

        population = mutate(population, mutateP)

        stats = getFitnessStats(population, items, bag.size)

        print("Generation: " + str(i) + ", Best fitness: " + str(stats[0]) + ", Average fitness: " + str(stats[1]))

        if stats[0] > bestFitness:
            bestFitness = stats[0]
            mostFitGenotype = stats[2]

        if stats[1] > bestAverageFitnessForPop:
            nonImprovingGens = 0
            bestAverageFitnessForPop = stats[1]
        else:
            nonImprovingGens = nonImprovingGens + 1

        if nonImprovingGens > convergenceThreshold:
            break

    fillBag(bag, mostFitGenotype, items)

    return(bag.value, i)

bagSize = 9

fitnessValue = fillBag(bagSize, items)
populationAfterExecution = fillByGenetics(bagSize, items)

print("--- Final Results ---")
print("Value: " + str(fitnessValue))
print("Total: " + str(populationAfterExecution))

