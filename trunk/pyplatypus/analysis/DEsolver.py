from __future__ import division
import numpy as np
import numpy.random as npr
import numpy.testing as npt


def diffevol(func, limits, args=(), DEstrategy=None,
             max_iterations=1000, popsize=20,
             tol=0.01, km=0.7, recomb=0.5, seed=None,
             progress=None):
    """
    A differential evolution minimizer - a stochastic way of minimizing functions.
    It does not use gradient methods to find the minimium, and can search large areas
    of candidate space, but often requires large numbers of function evaluations.   

    The algorithm is originally due to Storn and Price:
    http://www1.icsi.berkeley.edu/~storn/code.html

    http://en.wikipedia.org/wiki/Differential_evolution

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        `f(x, *args)`, where `x` is the argument in the form of a 1-D array
        and `args` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    limits: 2-D ndarray
        lower and upper limits for the optimizing argument of func. Must have
        shape (2, len(x))        
    args : tuple, optional
        Any additional fixed parameters needed to completely
        specify the objective function.
    DEstrategy : optional
        The differential evolution strategy to use.
    max_iterations: int, optional
        The maximum number of times the entire population is evolved
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        popsize * len(x) individuals.
    tol : float, optional:
        When the mean of the population energies, multiplied by tol,
        divided by the standard deviation of the population energies is greater than 1
        the solving process terminates.
        i.e. mean(pop) * tol / stdev(pop) > 1
    km : float, optional:
        The mutation constant, should be in the range [0, 1].
    recomb : float, optional:
        The recombination constant, should be in the range [0, 1].
    seed : float, optional:
        Seeds the random number generator for repeatable minimizations.
    progress : callable, optional:
        A function to follow the progress of the minimization.
        It has the signature: f(iteration, convergence, best_energy, *args)

    Returns
    -------
    xmin : ndarray
        The point where the lowest function value was found.
    Jmin : float
        The objective function value at `xmin`.
    """

    solver = DEsolver(func, limits, args=args, DEstrategy=DEstrategy,
                      max_iterations=max_iterations, popsize=popsize,
                      tol=tol, km=km, recomb=recomb, seed=seed,
                      progress=progress)

    x0, Jmin = solver.solve()
    return x0, Jmin


class DEsolver(object):

    """
    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        `f(x, *args)`, where `x` is the argument in the form of a 1-D array
        and `args` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    limits: 2-D ndarray
        lower and upper limits for the optimizing argument of func. Must have
        shape (2, len(x))        
    args : tuple, optional
        Any additional fixed parameters needed to completely
        specify the objective function.
    DEstrategy : str, optional
        The differential evolution strategy to use.
    max_iterations: int, optional
        The maximum number of times the entire population is evolved
    popsize : int, optional
        A multiplier for setting the total population size.  The population has
        popsize * len(x) individuals.
    tol : float, optional:
#         When abs((max(population_energies) - min(population_energies)) / min(population_energies)) < tol
#         the fit will stop.        
        When the mean of the population energies, multiplied by tol,
        divided by the standard deviation of the population energies is greater than 1
        the solving process terminates.
        i.e. mean(pop) * tol / stdev(pop) > 1
    km : float, optional:
        The mutation constant, should be in the range [0, 1].
    recomb : float, optional:
        The recombination constant, should be in the range [0, 1].
    seed : int, optional:
        Seed initializing the pseudo-random number generator. Can be an integer, an array
        (or other sequence) of integers of any length, or None (the default). If you use
        the seed you will get repeatable minimizations.
    progress : callable, optional:
        A function to follow the progress of the minimization.
        It has the signature: f(iteration, convergence, Jmin, *args)
    """

    def __init__(self, func, limits, args=(),
                 DEstrategy=None, max_iterations=1000, popsize=20,
                 tol=0.01, km=0.7, recomb=0.5, seed=None,
                 progress=None):

        if DEstrategy is not None:
            self.DEstrategy = getattr(DEsolver, DEstrategy)
        else: 
            self.DEstrategy = getattr(DEsolver, 'Best1Bin')
                
        self.progress = progress
        self.max_iterations = max_iterations
        self.tol = tol
        self.scale = km
        self.crossOverProbability = recomb

        self.func = func
        self.args = args
        self.limits = limits
        self.parameter_count = np.size(self.limits, 1)
        self.population_size = popsize * self.parameter_count

        self.RNG = npr.RandomState()
        self.RNG.seed(seed)

        self.population = self.RNG.rand(
            popsize *
            self.parameter_count,
            self.parameter_count)

        self.population_energies = np.ones(
            popsize * self.parameter_count) * 1.e300

    def solve(self):
        """
        Returns
        -------
        xmin : ndarray
            The point where the lowest function value was found.
        Jmin : float
            The objective function value at `xmin`.
        """

        # calculate energies to start with
        for index, candidate in enumerate(self.population):
            params = self.__scale_parameters(candidate)
            self.population_energies[
                index] = self.func(
                params,
                *self.args)

        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

        # do the optimisation.
        for iteration in xrange(self.max_iterations):

            for candidate in xrange(self.population_size):
                trial = self.DEstrategy(self, candidate)
                self.__ensure_constraint(trial)
                params = self.__scale_parameters(trial)
                energy = self.func(params, *self.args)

                if energy < self.population_energies[candidate]:
                    self.population[candidate] = trial
                    self.population_energies[candidate] = energy

                    if energy < self.population_energies[0]:
                        self.population_energies[0] = energy
                        self.population[0] = trial

            # stop when the fractional s.d. of the population is less than tol
            # of the mean energy
            convergence = np.std(self.population_energies) / \
                np.mean(self.population_energies)
#             convergence = np.abs((np.max(self.population_energies) - self.population_energies[0]) /
#                                 self.population_energies[0])
            
            if self.progress:
                should_continue = self.progress(
                    iteration,
                    convergence,
                    self.population_energies[0],
                    *self.args)
                if should_continue is False:
                    convergence = self.tol - 1

            if convergence < self.tol:
                self.convergence = convergence
                break

        return (
            self.__scale_parameters(
                self.population[0]), self.population_energies[0]
        )

    def __scale_parameters(self, trial):
        return (
            0.5 * (self.limits[0] + self.limits[1]) +
            (trial - 0.5) * np.fabs(self.limits[0] - self.limits[1])
        )

    def __ensure_constraint(self, trial):
        for index, param in enumerate(trial):
            if param > 1 or param < 0:
                trial[index] = self.RNG.rand()

    def Best1Bin(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 0, 0, 0)
        n = self.RNG.randint(0, self.parameter_count)
        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameter_count - 1:
                trial[n] = self.population[0, n] + self.scale * \
                    (self.population[r1, n] - self.population[r2, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def Best1Exp(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 0, 0, 0)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0
        while i < self.parameter_count and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.population[0, n] + self.scale * \
                (self.population[r1, n] - self.population[r2, n])
            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def Rand1Exp(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 1, 0, 0)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.population[r1, n] + self.scale * \
                (self.population[r2, n] - self.population[r3, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def RandToBest1Exp(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 0, 0, 0)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count and self.RNG.rand() < self.crossOverProbability:
            trial[n] += self.scale * (self.population[0, n] - trial[n]) + \
                self.scale * \
                (self.population[r1, n]
                 - self.population[r2, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def Best2Exp(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 1, 1, 0)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.population[0, n]
            + self.scale * (self.population[r1, n]
                            + self.population[r2, n]
                            - self.population[r3, n]
                            - self.population[r4, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def Rand2Exp(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 1, 1, 1)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count and self.RNG.rand() < self.crossOverProbability:
            trial[n] = self.population[r1, n]
            + self.scale * (self.population[r2, n]
                            + self.population[r3, n]
                            - self.population[r4, n]
                            - self.population[r5, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def RandToBest1Bin(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 0, 0, 0)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameter_count - 1:
                trial[n] += self.scale * (self.population[0, n] - trial[n])
                + self.scale * \
                    (self.population[r1, n] - self.population[r2, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def Best2Bin(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 1, 1, 0)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameter_count - 1:
                trial[n] = self.population[0, n]
                + self.scale * (self.population[r1, n]
                                + self.population[r2, n]
                                - self.population[r3, n]
                                - self.population[r4, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def Rand2Bin(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 1, 1, 1)

        n = self.RNG.randint(0, self.parameter_count)

        trial = np.copy(self.population[candidate])
        i = 0

        while i < self.parameter_count:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameter_count - 1:
                trial[n] = self.population[r1, n]
                + self.scale * (self.population[r2, n]
                                + self.population[r3, n]
                                - self.population[r4, n]
                                - self.population[r5, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def Rand1Bin(self, candidate):
        r1, r2, r3, r4, r5 = self.select_samples(candidate, 1, 1, 1, 0, 0)
        
        n = self.RNG.randint(0, self.parameter_count)
        
        trial = np.copy(self.population[candidate])
        i = 0
        while i < self.parameter_count:
            if self.RNG.rand() < self.crossOverProbability or i == self.parameter_count - 1:
                trial[n] = self.population[r1, n]
                + self.scale * (self.population[r2, n]
                                - self.population[r3, n])

            n = (n + 1) % self.parameter_count
            i += 1

        return trial

    def select_samples(self, candidate, r1, r2, r3, r4, r5):
        if r1:
            while True:
                r1 = self.RNG.randint(0, self.population_size)
                if r1 != candidate:
                    break
        if r2:
            while True:
                r2 = self.RNG.randint(0, self.population_size)
                if r2 != candidate and r1 != r2:
                    break
        if r3:
            while True:
                r3 = self.RNG.randint(0, self.population_size)
                if r3 != candidate and r3 != r2 and r3 != r1:
                    break
        if r4:
            while True:
                r4 = self.RNG.randint(0, self.population_size)
                if r4 != candidate and r4 != r3 and r4 != r2 and r4 != r1:
                    break
        if r5:
            while True:
                r5 = self.RNG.randint(0, self.population_size)
                if r5 != candidate and r5 != r4 and r5 != r3 and r5 != r2 and r5 != r1:
                    break

        return r1, r2, r3, r4, r5

if __name__ == "__main__":
    # minimum expected at ~-0.195
    func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    limits = np.array([[-3], [3]])
    solver = DEsolver(func, limits, tol=1e-2,
                          popsize=40, km=0.6, recomb=0.9, DEstrategy='Best1Bin')
    xmin, Jmin = solver.solve()
    print xmin, Jmin, solver.population_energies
    npt.assert_almost_equal(Jmin, func(xmin))
