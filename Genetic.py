import numpy as np
import random

class FeatureSet:

    def __init__(self, featureNum):
        self.features = list(np.random.choice([0,1], size=(featureNum)))
        self.fitness = -1

    def __eq__(self, featureSet):
        return self.features == featureSet.features

    def __str__(self):
        info = ''
        for num in range(len(self.features)):
            info += str(self.features[num])
        return info

class FeatureSelector:

    def __init__(self):
        self.popLen = 6
        self.featureNum = 13

    # take classifier and data too
    def ga(self, generations, clf, data):

        population = self.initializePopulation(self.popLen, self.featureNum)
        maxFitness = 0

        #for x in population:
        #    print(str(x))

        for gen in range(generations):

            print( 'Generation: ' + str(gen))
            #pass classifier to fitness
            population = self.fitness(population, clf, data)
            population = self.selection(population)
            currentMax = population[0].fitness
            if (currentMax>maxFitness):
                maxFitness = currentMax
            population = self.crossover(population)
            population = self.mutation(population)

            #Termination condition of  thealgorithm is when the maximum number of iteration reaches.
            """typical criteria areused such as a number of iterations, acceptable results or
                a fixed number of last generations without changingthe fitness value."""

        print("Max fitness: ", maxFitness)
            

            
    def initializePopulation(self, popLen, featureNum):
        population = []
        
        while len(population) < popLen:
            individual = FeatureSet(featureNum)
            if individual not in population:
                population.append(individual)

        return population
    # take classifier and dataset
    def fitness(self,population, clf, data):

        for featureSet in population:
            dataCopy = data.makeCopy()
            dataCopy.adjustFeatures(featureSet)
            print("Everything up to here happening")
            clf.train(dataCopy)
            featureSet.fitness = clf.getPerformance()
            print("Training done. Fitness: ",featureSet.fitness)
            # fitness= wA* TPR + wB * FPR + wC ->
            #In our study we set Wa to 40% and Wb to 50%  and  Wc  10%  which cause experiments to an
            #optimized result of high TP, low FP with small subset of selected features. 
            
        return population

    def selection(self,population):
        """Elitism replacement.  To  maintain  the  good  solutionsof each generation that may be lost during the evolu-tionary  process
        by  crossover  and  mutation  operators,we  use  the  elitism  replacement  technique """
        
        population = sorted(population, key=lambda featureSet: featureSet.fitness, reverse=True)
        population = population[:int(0.5 *len(population))]
        
        return population

    #test
    def allEqual(self,population):
        return all(individual == population[0] for individual in population)
    def crossover(self, population):

        offspring = []

        while len(offspring) < self.popLen/2:

            if self.allEqual(population):
                print("Here would get stuck")
                break

            while True:
                parent, parent2 = random.sample(population,2)
                if parent != parent2:
                    break

            featureNum = len(parent.features)
            child = FeatureSet(featureNum)
            child2 = FeatureSet(featureNum)

            split = random.randint(0, featureNum)
            #print(parent.features[0:split], (parent2.features[split:featureNum]))
            #print(parent2.features[0:split], (parent.features[split:featureNum]))
            child.features = parent.features[0:split] + (parent2.features[split:featureNum])
            child2.features = parent2.features[0:split] + (parent.features[split:featureNum])
            #print("Parentstog:\n", parent.features[0:split] + (parent2.features[split:featureNum]),"\n",parent2.features[0:split] + (parent.features[split:featureNum]))
            #print("Children: ", child,child2)

            offspring.append(child)
            offspring.append(child2)
            
        population.extend(offspring)

        if len(population) > self.popLen:
           population = population[:-1]

        #population.extend(offspring)
        """for x in population:
            print(str(x))"""

        return population

    def mutation(self, population):

        for featureSet in population:
            for index in range(len(featureSet.features)):
                if random.uniform(0.0, 1.0) <= 0.2:
                    featureSet.features[index] = np.random.choice([0,1])
        """print("\n")
        for x in population:
            print(str(x))"""
            
        return population
        



#selector = FeatureSelector()
#selector.ga(1)
