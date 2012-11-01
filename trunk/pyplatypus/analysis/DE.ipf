#pragma rtGlobals=3		// Use modern global access method.

//A Differential Optimisation implentation for IGOR.
//Copyright Andrew Nelson and ANSTO 2012.
//The original differential evolution ideas were developed by Storn and Price.
//http://www1.icsi.berkeley.edu/~storn/code.html
//
//set the DEoptim function for useage details.

	Structure DEoptimiser
	Funcref energyProtoType ef
	Funcref deStrategyPrototype de
	variable parameterCount
	variable scale
	variable crossOverProbability
	variable maxIterations
	variable populationSize
	variable tol
	variable iterations
	
	//population has dimensions [parameterCount][populationSize]
	Wave population
	Wave trial
	Wave scaledTrial
	Wave populationEnergies
	Wave pwave
	
	//limits has dimensions [parameterCount][2]
	Wave limits
	Wave bestSolution
	EndStructure

Function energyPrototype(pwave, xw)
	Wave pwave, xw
	print "For some reason the prototype energy function got called"

End

Function deStrategyPrototype(s, candidate)
	Struct DEoptimiser &s
	variable candidate
End

Function DEoptim(energyfunctionStr, limits, pwave, [scale, crossOverProbability, popsize, maxIterations, tol, DEStrategy])
	string energyfunctionStr
	Wave limits, pwave
	variable scale, crossoverProbability, popsize, maxIterations, tol
	String DEstrategy
	//Function that tries to find the global minimum of a user supplied energy function. If you wish to find a maximum of a user supplied function
	//then multiply the return value of your energy function by -1.
	
	//energyfunctionstr = string giving the name of the energy function.  It has the signature of the energyPrototype function. 
	//						The first wave passed to this function is 'pwave'.	pwave is unaltered during the optimisation process.
	//						 Use it to pass in extra information to the energy function.  You supply pwave to the DEoptim function.
	//						The second wave passed to this function are the independent variables which are being optimised.  This is supplied by the DEoptim
	//						function itself. This wave will have length dimsize(limits, 0).
	//						If you return NaN or INF the optimisation will terminate.
	//
	//limits = wave containing the lower and upper limits for the independent variables. This wave has dimensions [N][2], where N is the number of independent
	//			variables. The first column contains the lower limit, the second column contains the upper limit. The second wave supplied to energy function should
	//			 expect to receive a wave that is N parameters long.
	//
	//pwave = wave containing extra information to pass to the energyfunction. It is the first wave supplied to the energy function and is unaltered during the optimisation
	//			process.
	//
	
	if(paramisdefault(scale))
		scale = 0.7
	endif
	if(paramisdefault(crossOverProbability))
		scale = 0.5
	endif
	if(paramisdefault(popsize))
		popsize = 20
	endif
	if(paramisdefault(maxIterations))
		maxIterations = 1000
	endif
	if(paramisdefault(tol))
		tol = 0.01
	endif
	
	if(paramisdefault(DEStrategy))
		DEStrategy = "Best1Bin"
	endif

	Struct DEoptimiser s
	s.scale = scale
	s.crossOverProbability = crossOverProbability
	s.maxIterations = 1000
	s.tol = tol
	Funcref deStrategyPrototype s.de = $DEstrategy
	Funcref energyPrototype s.ef = $energyFunctionStr
	Wave s.limits = limits
	Wave s.pwave = pwave
	s.parameterCount = dimsize(limits, 0)
	s.populationSize = dimsize(limits, 0) * popsize
	
	//initialise population
	make/n=(s.parameterCount, s.populationSize)/free/d population
	Wave s.population = population
	population = abs(enoise(1, 2))
	
	make/n=(s.parameterCount)/d/free bestSolution, trial, scaledTrial
	Wave s.bestSolution = bestSolution
	Wave s.trial = trial
	Wave s.scaledTrial = scaledTrial
	make/n=(s.populationSize)/d/free populationEnergies
	Wave s.populationEnergies = populationEnergies
	
	s.population = abs(enoise(1, 2))
	s.bestSolution[] = s.population[p][0]
	s.populationEnergies = 1.e300
	
	solve(s)
	//return best parameters and best energy
	make/n=(s.parameterCount)/d/o W_Extremum = 0
	variable/g V_min = s.populationenergies[0]
	scale_parameters(s, s.bestSolution, W_Extremum)
	variable/g V_OptNumIters = s.iterations
End

Function solve(s)
	//minimize the energy function.
	Struct DEoptimiser &s
	
	Wave population = s.population
	Wave populationEnergies = s.populationEnergies
	Wave bestSolution =  s.bestSolution
	Wave trial = s.trial
	Wave scaledTrial = s.scaledTrial
	Wave pwave = s.pwave
	Funcref energyProtoType ef = s.ef
	Funcref deStrategyPrototype de = s.de

	variable iteration, candidate, kk, convergence = 0, minval, minloc, energy

	//calculate energies to start with
	for(candidate = 0 ; candidate < s.populationSize ; candidate += 1)
		trial[] = s.population[p][candidate]
		scale_parameters(s, trial, scaledTrial)
		populationEnergies[candidate] = ef(pwave, scaledTrial)
	endfor
	//find the minimum energy and put in best solution place
	Wavestats/z/q/M=1 populationenergies
	populationEnergies[V_minloc] = populationenergies[0]
	populationEnergies[0] = V_min
	trial[] = population[p][V_minloc]
	population[][0] = population[p][V_minloc]
	population[][V_minloc] = trial[p]        
	bestSolution[] = population[p][0]
	
	//do the optimisation.
	for(iteration = 0 ; iteration < s.maxIterations && convergence < 1 ; iteration+=1)
		for(candidate = 0 ; candidate < s.populationSize ; candidate += 1)
			de(s, candidate)
			ensure_constraint(trial)
			scale_parameters(s, trial, scaledTrial)

			energy = ef(pwave, scaledTrial)
                	
                	//abort if INF or NaN is return from energy function
                	if(numtype(energy))
                		abort "Energy function returned NaN or INF"
                	endif
                	
                	
			if(energy < populationenergies[candidate])
				population[][candidate] = trial[p]
				populationenergies[candidate] = energy
                    
				if(energy < populationenergies[0])
					populationenergies[0] = energy
					population[][0] = trial[p]
					bestSolution[] = trial[p]
				endif
			endif
		endfor
		//stop when the fractional s.d. of the population is less than tol of the mean energy
		wavestats/q/z populationenergies
		convergence = abs(V_avg) * s.tol / V_sdev 
	endfor
	s.iterations = iteration
End

Function Best1Bin(s, candidate)
	Struct DEoptimiser &s
	variable candidate
	variable r1 = 1, r2 = 1, r3, r4, r5
	variable n, i
	
	Wave  population = s.population
	Wave  trial = s.trial
	Wave bestSolution = s.bestSolution
	
	selectSamples(s.populationSize, candidate, r1, r2, r3, r4, r5)
	n = randint(s.parameterCount)	 	
	trial[] = population[p][candidate]
	i = 0
	do
		if(abs(enoise(1, 2)) < s.crossOverProbability ||  i == s.parameterCount - 1)
			trial[n] = bestSolution[n] + s.scale * (population[n][r1] - population[n][r2])
		endif
		n = mod(n + 1, s.parameterCount)
		i += 1
	while (i < s.parameterCount)
End

Function Best1Exp(s, candidate)
	Struct DEoptimiser &s
	variable candidate
	variable r1 = 1, r2 = 1, r3, r4, r5
	variable n, i
	
	Wave  population = s.population
	Wave  trial = s.trial
	Wave bestSolution = s.bestSolution
	
	selectSamples(s.populationSize, candidate, r1, r2, r3, r4, r5)
	n = randint(s.parameterCount)	 	
	trial[] = population[p][candidate]
	i = 0
	
	for(i = 0 ; i < s.parameterCount && abs(enoise(1,2)) < s.crossOverProbability ; i += 1)
		trial[n] = bestSolution[n] + s.scale * (population[n][r1] - population[n][r2])	
		n = mod(n + 1, s.parameterCount)
	endfor
End

Function Rand1Exp(s, candidate)
	Struct DEoptimiser &s
	variable candidate
	variable r1 = 1, r2 = 1, r3 = 1, r4, r5
	variable n, i
	
	Wave  population = s.population
	Wave  trial = s.trial
	Wave bestSolution = s.bestSolution
	
	selectSamples(s.populationSize, candidate, r1, r2, r3, r4, r5)
	n = randint(s.parameterCount)	 	
	trial[] = population[p][candidate]
	i = 0
	
	for(i = 0 ; i < s.parameterCount && abs(enoise(1,2)) < s.crossOverProbability ; i += 1)
		trial[n] = population[n][r1] + s.scale * (population[n][r2] - population[n][r3])	
		n = mod(n + 1, s.parameterCount)
	endfor
End

//    def RandToBest1Exp(self, candidate):
//        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 0, 0, 0)
//
//        n = self.RNG.randint(0, self.parameterCount)
//
//        trial = np.copy(self.population[candidate])
//        i = 0
//        
//        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
//            trial[n] += self.scale * (self.bestSolution[n] - trial[n]) + self.scale * (self.population[r1, n] - self.population[r2, n])
//            
//            n = (n + 1) % self.parameterCount
//            i += 1
//        
//        return trial
//
//    def Best2Exp(self, candidate):
//        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 0)
//
//        n = self.RNG.randint(0, self.parameterCount)
//
//        trial = np.copy(self.population[candidate])
//        i = 0
//
//        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
//            trial[n] = self.bestSolution[n]
//            + self.scale * (self.population[r1, n]
//            + self.population[r2, n]
//            - self.population[r3, n]
//            - self.population[r4, n])
//
//            n = (n + 1) % self.parameterCount
//            i += 1
//
//        return trial
//
//    def Rand2Exp(self, candidate):
//        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 1)
//
//        n = self.RNG.randint(0, self.parameterCount)
//
//        trial = np.copy(self.population[candidate])
//        i = 0
//
//        while i < self.parameterCount and self.RNG.rand() < self.crossOverProbability:
//            trial[n] = self.population[r1, n]
//            + self.scale * (self.population[r2, n] 
//            + self.population[r3, n] 
//            - self.population[r4, n] 
//            - self.population[r5, n]) 
//
//            n = (n + 1) % self.parameterCount
//            i += 1
//
//        return trial
//
//    def RandToBest1Bin(self, candidate):
//        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 0, 0, 0)
//
//        n = self.RNG.randint(0, self.parameterCount)
//
//        trial = np.copy(self.population[candidate])
//        i = 0
//
//        while i < self.parameterCount:
//            if self.RNG.rand() < self.crossOverProbability or i == self.parameterCount - 1:
//                trial[n] += self.scale * (self.bestSolution[n] - trial[n])
//                + self.scale * (self.population[r1, n] - self.population[r2, n])
//
//            n = (n + 1) % self.parameterCount
//            i += 1
//
//        return trial
//        
//    def Best2Bin(self, candidate):
//        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 0)
//
//        n = self.RNG.randint(0, self.parameterCount)
//
//        trial = np.copy(self.population[candidate])
//        i = 0
//
//        while i < self.parameterCount:
//            if self.RNG.rand() < self.crossOverProbability or i == self.parameterCount - 1:
//                trial[n] = self.bestSolution[n]
//                + self.scale * (self.population[r1, n]
//                + self.population[r2, n]
//                -  self.population[r3, n]
//                -  self.population[r4, n])
//
//            n = (n + 1) % self.parameterCount
//            i += 1
//
//        return trial
//        
//    def Rand2Bin(self, candidate):
//        r1,r2,r3,r4,r5 = self.SelectSamples(candidate, 1, 1, 1, 1, 1)
//
//        n = self.RNG.randint(0, self.parameterCount)
//
//        trial = np.copy(self.population[candidate])
//        i = 0
//
//        while i < self.parameterCount:
//            if self.RNG.rand() < self.crossOverProbability or i == self.parameterCount - 1:
//                trial[n] = self.population[r1, n]
//                + self.scale * (self.population[r2, n]
//                + self.population[r3, n]
//                -  self.population[r4, n]
//                -  self.population[r5, n])
//
//            n = (n + 1) % self.parameterCount
//            i += 1
//
//        return trial

Function Rand1Bin(s, candidate)
	Struct DEoptimiser &s
	variable candidate
	variable r1 = 1, r2 = 1, r3 = 1, r4, r5
	variable n, i
	
	Wave  population = s.population
	Wave  trial = s.trial
	Wave bestSolution = s.bestSolution
	
	selectSamples(s.populationSize, candidate, r1, r2, r3, r4, r5)
	n = randint(s.parameterCount)	 	
	trial[] = population[p][candidate]
	i = 0
	
	for(i = 0 ; i < s.parameterCount ; i += 1)
		if (abs(enoise(1,2)) < s.crossOverProbability || i == s.parameterCount - 1)
			trial[n] = population[n][r1] + s.scale * (population[n][r2] - population[n][r3])
		endif
		n = mod(n + 1, s.parameterCount)
	endfor
End
        
Function SelectSamples(popsize, candidate, r1, r2, r3, r4, r5)
	variable popsize, candidate, &r1, &r2, &r3, &r4, &r5
       
	if (r1)
		do
			r1 = randInt(popsize)
		while (r1 == candidate)
	endif
	if (r2)
		do
			r2 = randInt(popsize)
		while(r2 == r1 || r2 == candidate)
	endif
	if (r3)
		do 
			r3 = randInt(popsize)
		while (r3 == r2 || r3 == r1 || r3 == candidate)
	endif
	if (r4)
		do
			r4 = randInt(popsize)
		while (r4 == r3 || r4 == r2 || r4 == r1 || r4 == candidate)
	endif
	if (r5)
		do          
			r5 = randInt(popsize)
		while (r5 == r4  || r5 == r3 || r5 == r2 || r5 == r1 || r5 == candidate)
	endif
End
        
        
Threadsafe  Function randInt(val)
	variable val
	//returns a random integer in the range [0, val), val IS NOT INCLUDED
	return floor(abs(enoise(val, 2)))
End

Function scale_parameters(s, trial, scaledTrial)
	//the values in the population are scaled between 0 and 1, representing what fraction they lie
	//on the interval between the lower and upper limit. This function scales the fraction to the actual 
	//value.
	Struct DEoptimiser &s
	Wave trial, scaledTrial
	
	Wave limits =  s.limits
	scaledTrial = 0.5 * (limits[p][0] + limits[p][1]) + (trial[p] - 0.5) * abs(limits[p][0] - limits[p][1])
End

Function ensure_constraint(trial)
	//all values in the population should lie between 0 and 1
	Wave trial
	variable ii
	for(ii = 0 ; ii < numpnts(trial) ; ii += 1)
		if(trial[ii] > 1 || trial[ii] < 0)
			trial[ii] = abs(enoise(1, 2))
		endif
	endfor                
End