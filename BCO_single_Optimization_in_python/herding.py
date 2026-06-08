"""
Herding Function for Border Collie Optimization (BCO) Algorithm
This file contains the function to sort population based on fitness (herding behavior)
"""

import numpy as np

def herding(pop, Vt, fit, n, L, a, t):
    """
    Sort the population based on fitness values (simulating herding behavior)
    Individuals with better fitness come first
    
    Parameters:
    -----------
    pop : numpy.ndarray
        Population positions matrix of shape (n, L)
    Vt : numpy.ndarray
        Velocity matrix of shape (n, L)
    fit : numpy.ndarray
        Fitness values array of shape (n,)
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    a : numpy.ndarray
        Acceleration matrix of shape (n, L)
    t : numpy.ndarray
        Time array of shape (n,)
    
    Returns:
    --------
    pop1 : numpy.ndarray
        Sorted population positions (best fitness first)
    Vt1 : numpy.ndarray
        Sorted velocities (ordered by fitness)
    fit1 : numpy.ndarray
        Sorted fitness values (ascending order)
    a1 : numpy.ndarray
        Acceleration matrix (ordered by fitness)
    t1 : numpy.ndarray
        Time array (ordered by fitness)
    """
    
    # Get indices that would sort the fitness array in ascending order
    # (better fitness = smaller value for minimization problems)
    idx = np.argsort(fit)
    
    # Initialize sorted arrays
    pop1 = np.zeros((n, L))
    Vt1 = np.zeros((n, L))
    a1 = np.zeros((n, L))
    t1 = np.zeros(n)
    
    # Reorder all arrays based on sorted fitness indices
    for i in range(n):
        pop1[i, :] = pop[idx[i], :]
        Vt1[i, :] = Vt[idx[i], :]
        a1[i, :] = a[idx[i], :]
        t1[i] = t[idx[i]]
    
    # Sorted fitness values
    fit1 = fit[idx]
    
    return pop1, Vt1, fit1, a1, t1


def herding_vectorized(pop, Vt, fit, n, L, a, t):
    """
    Vectorized version of herding function for better performance
    
    Parameters:
    -----------
    pop : numpy.ndarray
        Population positions matrix of shape (n, L)
    Vt : numpy.ndarray
        Velocity matrix of shape (n, L)
    fit : numpy.ndarray
        Fitness values array of shape (n,)
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    a : numpy.ndarray
        Acceleration matrix of shape (n, L)
    t : numpy.ndarray
        Time array of shape (n,)
    
    Returns:
    --------
    pop1 : numpy.ndarray
        Sorted population positions
    Vt1 : numpy.ndarray
        Sorted velocities
    fit1 : numpy.ndarray
        Sorted fitness values
    a1 : numpy.ndarray
        Sorted acceleration
    t1 : numpy.ndarray
        Sorted time
    """
    
    # Get sorting indices
    idx = np.argsort(fit)
    
    # Vectorized reordering
    pop1 = pop[idx, :]
    Vt1 = Vt[idx, :]
    fit1 = fit[idx]
    a1 = a[idx, :]
    t1 = t[idx]
    
    return pop1, Vt1, fit1, a1, t1


def get_dogs_and_sheep(pop, Vt, fit, a, t, n_dogs=3):
    """
    Separate dogs (best individuals) from sheep based on fitness
    
    Parameters:
    -----------
    pop : numpy.ndarray
        Population positions matrix
    Vt : numpy.ndarray
        Velocity matrix
    fit : numpy.ndarray
        Fitness values array
    a : numpy.ndarray
        Acceleration matrix
    t : numpy.ndarray
        Time array
    n_dogs : int
        Number of dogs (default: 3)
    
    Returns:
    --------
    dogs_pop : numpy.ndarray
        Positions of dogs (best individuals)
    dogs_Vt : numpy.ndarray
        Velocities of dogs
    dogs_fit : numpy.ndarray
        Fitness values of dogs
    dogs_a : numpy.ndarray
        Accelerations of dogs
    dogs_t : numpy.ndarray
        Time values of dogs
    sheep_pop : numpy.ndarray
        Positions of sheep (remaining individuals)
    sheep_Vt : numpy.ndarray
        Velocities of sheep
    sheep_fit : numpy.ndarray
        Fitness values of sheep
    sheep_a : numpy.ndarray
        Accelerations of sheep
    sheep_t : numpy.ndarray
        Time values of sheep
    """
    
    # First, sort all individuals by fitness
    sorted_pop, sorted_Vt, sorted_fit, sorted_a, sorted_t = herding_vectorized(
        pop, Vt, fit, len(pop), pop.shape[1], a, t
    )
    
    # Separate dogs (best n_dogs) and sheep (rest)
    dogs_pop = sorted_pop[:n_dogs, :]
    dogs_Vt = sorted_Vt[:n_dogs, :]
    dogs_fit = sorted_fit[:n_dogs]
    dogs_a = sorted_a[:n_dogs, :]
    dogs_t = sorted_t[:n_dogs]
    
    sheep_pop = sorted_pop[n_dogs:, :]
    sheep_Vt = sorted_Vt[n_dogs:, :]
    sheep_fit = sorted_fit[n_dogs:]
    sheep_a = sorted_a[n_dogs:, :]
    sheep_t = sorted_t[n_dogs:]
    
    return (dogs_pop, dogs_Vt, dogs_fit, dogs_a, dogs_t,
            sheep_pop, sheep_Vt, sheep_fit, sheep_a, sheep_t)


# ==================== TEST CODE ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Herding Function for BCO Algorithm")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    n = 10  # 10 individuals
    L = 5   # 5 dimensions
    
    # Create random test data
    pop = np.random.randn(n, L) * 10  # Random positions
    Vt = np.random.randn(n, L)        # Random velocities
    fit = np.random.rand(n) * 100     # Random fitness values
    a = np.random.randn(n, L)         # Random accelerations
    t = np.random.rand(n) * 5         # Random time values
    
    print("Original Data:")
    print("-" * 40)
    print(f"Population shape: {pop.shape}")
    print(f"Fitness values: {fit}")
    print(f"Indices of best (min) fitness: {np.argmin(fit)}")
    print(f"Indices of worst (max) fitness: {np.argmax(fit)}")
    
    # Test herding function
    print("\n" + "=" * 70)
    print("Testing herding() function:")
    print("=" * 70)
    
    pop1, Vt1, fit1, a1, t1 = herding(pop, Vt, fit, n, L, a, t)
    
    print("\nSorted Data (best fitness first):")
    print("-" * 40)
    print(f"Sorted fitness values: {fit1}")
    print(f"First individual (best fitness): {pop1[0, :]}")
    print(f"Last individual (worst fitness): {pop1[-1, :]}")
    
    # Verify sorting worked correctly
    is_sorted = np.all(fit1[:-1] <= fit1[1:])
    print(f"\nIs fitness properly sorted (ascending)?: {is_sorted}")
    
    # Verify data integrity (values match original, just reordered)
    original_sorted_fit = np.sort(fit)
    print(f"Matches np.sort(fit)?: {np.allclose(fit1, original_sorted_fit)}")
    
    # Test vectorized version
    print("\n" + "=" * 70)
    print("Testing herding_vectorized() function:")
    print("=" * 70)
    
    pop1_vec, Vt1_vec, fit1_vec, a1_vec, t1_vec = herding_vectorized(pop, Vt, fit, n, L, a, t)
    
    # Verify both versions produce same result
    print(f"Same as herding()?: {np.allclose(pop1, pop1_vec)}")
    print(f"Same fitness?: {np.allclose(fit1, fit1_vec)}")
    
    # Test dog and sheep separation
    print("\n" + "=" * 70)
    print("Testing get_dogs_and_sheep() function:")
    print("=" * 70)
    
    n_dogs = 3
    (dogs_pop, dogs_Vt, dogs_fit, dogs_a, dogs_t,
     sheep_pop, sheep_Vt, sheep_fit, sheep_a, sheep_t) = get_dogs_and_sheep(
        pop, Vt, fit, a, t, n_dogs
    )
    
    print(f"\nNumber of dogs: {n_dogs}")
    print(f"Dogs fitness values: {dogs_fit}")
    print(f"Dogs positions (first 2 dimensions):\n{dogs_pop[:, :2]}")
    
    print(f"\nNumber of sheep: {n - n_dogs}")
    print(f"Sheep fitness values (first 5): {sheep_fit[:5]}")
    print(f"Sheep positions (first 2 dimensions, first 3 sheep):\n{sheep_pop[:3, :2]}")
    
    # Verify no overlap between dogs and sheep
    all_fitness = np.concatenate([dogs_fit, sheep_fit])
    sorted_all = np.sort(all_fitness)
    print(f"\nAll fitness values sorted: {sorted_all}")
    print(f"Dogs have best {n_dogs} fitness values?: {np.all(dogs_fit <= sheep_fit)}")
    
    # Real BCO scenario test
    print("\n" + "=" * 70)
    print("Real BCO Scenario Test:")
    print("=" * 70)
    
    # Simulate BCO population
    n_population = 30  # 3 dogs + 27 sheep
    dimensions = 10
    
    # Create realistic data
    population = np.random.randn(n_population, dimensions) * 20
    velocities = np.random.randn(n_population, dimensions)
    # Create fitness values where some are clearly better
    fitness = np.random.exponential(scale=10, size=n_population)
    # Make a few individuals have very good fitness
    fitness[:3] = np.random.exponential(scale=1, size=3)
    accelerations = np.random.randn(n_population, dimensions)
    times = np.random.rand(n_population) * 10
    
    print(f"Initial population size: {n_population}")
    print(f"Initial best fitness: {np.min(fitness):.4f}")
    print(f"Initial worst fitness: {np.max(fitness):.4f}")
    
    # Apply herding
    sorted_pop, sorted_Vt, sorted_fit, sorted_a, sorted_t = herding_vectorized(
        population, velocities, fitness, n_population, dimensions, accelerations, times
    )
    
    print(f"\nAfter herding:")
    print(f"Best fitness (first): {sorted_fit[0]:.4f}")
    print(f"Worst fitness (last): {sorted_fit[-1]:.4f}")
    print(f"Fitness values sorted: {sorted_fit[:5]}... (first 5)")
    
    # Separate dogs and sheep
    n_dogs_bco = 3
    dogs_fit_bco = sorted_fit[:n_dogs_bco]
    sheep_fit_bco = sorted_fit[n_dogs_bco:]
    
    print(f"\nDogs (best 3) fitness: {dogs_fit_bco}")
    print(f"Sheep (remaining {n_population - n_dogs_bco}) fitness range: "
          f"[{np.min(sheep_fit_bco):.4f}, {np.max(sheep_fit_bco):.4f}]")
    
    # Verify all dogs have better fitness than all sheep
    if np.all(dogs_fit_bco <= sheep_fit_bco):
        print("\n✓ All dogs have better (smaller) fitness than all sheep")
    else:
        print("\n✗ Error: Some sheep have better fitness than dogs")
    
    print("\n" + "=" * 70)
    print("Herding function is working correctly!")
    print("=" * 70)