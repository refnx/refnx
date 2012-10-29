#pragma rtGlobals=3		// Use modern global access method.



	Structure DEoptimiser
	Funcref energyProtoType ef
	Funcref deStrategyPrototype de
	variable parameterCount
	variable scale
	variable crossOverProbability
	variable maxIterations
	variable populationSize
	variable tol
	
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

Function energyPrototype(w, xw)
	Wave w, xw
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
	
	make/n=(s.parameterCount, s.populationSize)/free/d population
	Wave s.population = population
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
	make/n=(s.parameterCount)/d/o W_min = 0
	variable/g V_min = s.populationenergies[0]
	scale_parameters(s, s.bestSolution, W_min)
End

Function solve(s)
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
		convergence = V_avg * s.tol / V_sdev 
	endfor
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
	Struct DEoptimiser &s
	Wave trial, scaledTrial
	
	Wave limits =  s.limits
	scaledTrial = 0.5 * (limits[p][0] + limits[p][1]) + (trial[p] - 0.5) * abs(limits[p][0] - limits[p][1])
End

Function ensure_constraint(trial)
	Wave trial
	variable ii
	for(ii = 0 ; ii < numpnts(trial) ; ii += 1)
		if(trial[ii] > 1 || trial[ii] < 0)
			trial[ii] = abs(enoise(1, 2))
		endif
	endfor                
End