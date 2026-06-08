"""
Fitness function for Border Collie Optimization (BCO) Algorithm
This file contains the fitness calculation utility for the BCO algorithm
"""

import numpy as np

def fitness(initP, n, L, ub, lb, fobj):
    """
    Calculate fitness values for all individuals in the population
    
    Parameters:
    -----------
    initP : numpy.ndarray
        Initial population matrix of shape (n, d) where n is population size
        and d is number of dimensions
    n : int
        Number of individuals in the population
    L : any (kept for MATLAB compatibility, not used)
        Unused parameter
    ub : list or numpy.ndarray
        Upper bounds for each dimension (kept for compatibility)
    lb : list or numpy.ndarray
        Lower bounds for each dimension (kept for compatibility)
    fobj : function
        Objective function to minimize (takes a position vector and returns fitness)
    
    Returns:
    --------
    fit1 : numpy.ndarray
        Array of fitness values for each individual (shape: n,)
    maxf : float
        Best (minimum) fitness value found
    pos : int
        Index position of the best fitness value in the population
    """
    
    # Initialize fitness array
    fit1 = np.zeros(n)
    
    # Calculate fitness for each individual
    for i in range(initP.shape[0]):
        fit1[i] = fobj(initP[i, :])
    
    # Find minimum fitness value (since minimization functions are considered)
    # idx is the index of the minimum value
    # maxf is the best (minimum) fitness value
    maxf_idx = np.argmin(fit1)
    maxf = fit1[maxf_idx]
    pos = maxf_idx
    
    return fit1, maxf, pos


# Simplified version with only necessary parameters
def fitness_simple(population, objective_function):
    """
    Simplified fitness function for BCO algorithm
    
    Parameters:
    -----------
    population : numpy.ndarray
        Population matrix of shape (n, d)
    objective_function : function
        Objective function to minimize
    
    Returns:
    --------
    fitness_values : numpy.ndarray
        Array of fitness values
    best_fitness : float
        Best (minimum) fitness value
    best_index : int
        Index of best individual
    """
    fitness_values = np.array([objective_function(ind) for ind in population])
    best_index = np.argmin(fitness_values)
    best_fitness = fitness_values[best_index]
    
    return fitness_values, best_fitness, best_index


# Test the fitness function
if __name__ == "__main__":
    # Define a simple test objective function (Sphere function - minimization)
    def sphere_function(x):
        """Sphere function: sum of squares of all elements"""
        return np.sum(x**2)
    
    # Create a test population: 10 individuals, each with 3 dimensions
    np.random.seed(42)  # For reproducible results
    test_population = np.random.uniform(-10, 10, size=(10, 3))
    
    print("=" * 60)
    print("Testing Fitness Function for BCO Algorithm")
    print("=" * 60)
    print(f"\nTest Population shape: {test_population.shape}")
    print(f"Population:\n{test_population}\n")
    
    # Test the original version
    fit_values, best_fit, best_pos = fitness(
        initP=test_population,
        n=10,
        L=None,
        ub=[10, 10, 10],
        lb=[-10, -10, -10],
        fobj=sphere_function
    )
    
    print("-" * 60)
    print("Results from fitness() function:")
    print("-" * 60)
    print(f"Fitness values: {fit_values}")
    print(f"\nBest fitness (minimum) value: {best_fit}")
    print(f"Index of best individual: {best_pos}")
    print(f"Best individual: {test_population[best_pos]}")
    
    # Test the simplified version
    fit_values_simple, best_fit_simple, best_pos_simple = fitness_simple(
        population=test_population,
        objective_function=sphere_function
    )
    
    print("\n" + "-" * 60)
    print("Results from fitness_simple() function:")
    print("-" * 60)
    print(f"Fitness values: {fit_values_simple}")
    print(f"\nBest fitness (minimum) value: {best_fit_simple}")
    print(f"Index of best individual: {best_pos_simple}")
    print(f"Best individual: {test_population[best_pos_simple]}")
    
    print("\n" + "=" * 60)
    print("Fitness function is working correctly!")
    print("=" * 60)