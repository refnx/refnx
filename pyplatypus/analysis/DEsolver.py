from __future__ import division
import numpy as np
import numpy.random as npr
import math

class DEsolver(object):
    
    def __init__(self, limits, energy_function, args_tuple = (),
                    initial_params = None, DEstrategy = None, maxIterations = 1000,
                    popsize = 20, tol = 0.01, km = 0.7, recomb = 0.5, seed = None,
                    progress = None):

        if DEstrategy is not None:
            self.DEstrategy = DEstrategy
        else:
            self.DEstrategy = DEsolver.Best1Bin
        
        self.progress = progress
        self.maxIterations = maxIterations
        self.tol = tol
        self.scale = km
        self.crossOverProbability = recomb

        self.energy_function = energy_function
        self.args_tuple = args_tuple
        self.limits = limits
        self.parameterCount = np.size(self.limits, 1)
        self.populationSize = popsize * self.parameterCount

        self.RNG = npr.RandomState()
        self.RNG.seed(seed)

        self.population = self.RNG.rand(popsize * self.parameterCount, self.parameterCount)
        self.bestSolution = self.population[0]
        if initial_params is not None:
            self.population[0] = initial_params

        self.population_energies = np.ones(popsize * self.parameterCount) * 1.e300
        
    def solve(self):
        #calculate energies to start with
        for index, candidate in enumerate(self.population):
            params = self.__scale_parameters(candidate)
            self.population_energies[index] = self.energy_function(params, self.args_tuple)
        
        minval = np.argmin(self.population_energies)

        #put the lowest energy into the best solution position.
        self.population_energies[minval], self.population_energies[0] = self.population_energies[0], self.population_energies[minval]
        self.population[minval], self.population[0] = self.population[0], self.population[minval]
        
        #do the optimisation.
        for iteration in xrange(self.maxIterations):
                
            for candidate in xrange(self.populationSize):
                trial = self.DEstrategy(self, candidate)
                self.__ensure_constraint(trial)
                params = self.__scale_parameters(trial)

                energy = self.energy_function(params, self.args_tuple)
                
                if energy < self.population_energies[candidate]:
                    self.population[candidate] = trial
                    self.population_energies[candidate] = energy
                    
                    if energy < self.population_energies[0]:
                        self.population_energies[0] = energy
                        self.population[0] = trial
            
            #stop when the fractional s.d. of the population is less than tol of the mean energy
            convergence = np.mean(self.population_energies) * self.tol / np.std(self.population_energies)

            if self.progress:
                should_continue = self.progress(iteration, convergence, self.population_energies[0], self.args_tuple)
                if should_continue is False:
                    convergence = 2
                    
            if convergence > 1:
                break
        
        return self.__scale_parameters(self.bestSolution), self.population_energies[0]
        
    def __scale_parameters(self, trial):
        return 0.5 * (self.limits[0] + self.limits[1]) + (trial - 0.5) * np.fabs(self.limits[0] - self.limits[1])
    
    def __ensure_constraint(self, trial):
        for index, param in enumerate(trial):
            if param > 1 or param < 0:
                trial[index] = self.RNG.rand()
    
    def Best1Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0
        
        while i < self.parameterCount: 
            if self.RNG.rand() < self.crossOverProbability or  i == self.parameterCount - 1:
                trial[n] = self.bestSolution[n] + self.scale * (self.population[r1, n] - self.population[r2, n])
        
            n = (n + 1) % self.parameterCount
            i += 1
    
        return trial

    def Best1Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1,1,0,0,0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0
        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.bestSolution[n] + self.scale * (self.population[r1, n] - self.population[r2, n])
            n = (n + 1) % self.parameterCount
            i += 1

        return trial
    
    def Rand1Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 0, 0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0
        
        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.population[r1, n] + self.scale * (self.population[r2, n] - self.population[r3, n])
            
            n = (n + 1) % self.parameterCount
            i += 1
        
        return trial
    
    def RandToBest1Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 0, 0, 0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0
        
        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
            trial[n] += self.scale * (self.bestSolution[n] - trial[n]) + self.scale * (self.population[r1, n] - self.population[r2, n])
            
            n = (n + 1) % self.parameterCount
            i += 1
        
        return trial

    def Best2Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.bestSolution[n]
            + self.scale * (self.population[r1, n]
            + self.population[r2, n]
            - self.population[r3, n]
            - self.population[r4, n])

            n = (n + 1) % self.parameterCount
            i += 1

        return trial

    def Rand2Exp(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 1)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.population[r1, n]
            + self.scale * (self.population[r2, n] 
            + self.population[r3, n] 
            - self.population[r4, n] 
            - self.population[r5, n]) 

            n = (n + 1) % self.parameterCount
            i += 1

        return trial

    def RandToBest1Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 0, 0, 0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameterCount:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameterCount - 1:
                trial[n] += self.scale * (self.bestSolution[n] - trial[n])
                + self.scale * (self.population[r1, n] - self.population[r2, n])

            n = (n + 1) % self.parameterCount
            i += 1

        return trial
        
    def Best2Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameterCount:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameterCount - 1:
                trial[n] = self.bestSolution[n]
                + self.scale * (self.population[r1, n]
                + self.population[r2, n]
                -  self.population[r3, n]
                -  self.population[r4, n])

            n = (n + 1) % self.parameterCount
            i += 1

        return trial
        
    def Rand2Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 1)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameterCount:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameterCount - 1:
                trial[n] = self.population[r1, n]
                + self.scale * (self.population[r2, n]
                + self.population[r3, n]
                -  self.population[r4, n]
                -  self.population[r5, n])

            n = (n + 1) % self.parameterCount
            i += 1

        return trial
        
    def Rand1Bin(self, candidate):
        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 0, 0)

        n = self.RNG.randint(0, self.parameterCount)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameterCount:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameterCount - 1:
                trial[n] = self.population[r1, n]
                + self.scale * (self.population[r2, n]
                - self.population[r3, n])

            n = (n + 1) % self.parameterCount
            i += 1

        return trial
        
    def SelectSamples(self, candidate, r1, r2, r3, r4, r5):
        if r1:
            while r1 == candidate:
                r1 = self.RNG.randint(0, self.populationSize)
        if r2:
            while (r2 == r1 or r2 == candidate):
                r2 = self.RNG.randint(0, self.populationSize)
        if r3:
            while (r3 == r2 or r3 == r1 or r3 == candidate):
                r3 = self.RNG.randint(0, self.populationSize)
        if r4:
            while (r4 == r3 or r4 == r2 or r4 == r1 or r4 == candidate):
                r4 = self.RNG.randint(0, self.populationSize)
        if r5:
            while (r5 == r4 or r5 == r3 or r5 == r2 or r5 == r1 or r5 == candidate):
                r5 = self.RNG.randint(0, self.populationSize)

        return r1, r2, r3, r4, r5