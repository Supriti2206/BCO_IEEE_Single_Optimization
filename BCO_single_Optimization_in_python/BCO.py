"""
Border Collie Optimization (BCO) Algorithm - Main Script
This file implements the complete BCO algorithm for single-objective optimization
"""

import numpy as np
from function_details import func_details
from generate import generate
from fitness import fitness
from herding import herding, herding_vectorized, get_dogs_and_sheep
from velocity_time_acceleration import update_velocity_time_acceleration
from position_update import update, update_vectorized
from check import check, check_with_logging


def border_collie_optimization(fname='func1', n=30, gen=200, verbose=True):
    """
    Border Collie Optimization (BCO) Algorithm
    
    Parameters:
    fname : str
        Name of the optimization function (e.g., 'func1', 'func2', ..., 'func19')
    n : int
        Population size (default: 30)
    gen : int
        Maximum number of iterations/generations (default: 200)
    verbose : bool
        If True, print progress information (default: True)
    
    Returns:
    fopt : float
        Best (optimal) fitness value found
    best_position : numpy.ndarray
        Position of the best solution
    fopt_history : numpy.ndarray
        History of best fitness values over generations
    """
    
    # Retrieve function details
    fobj, lb, ub, L = func_details(fname)
    
    print("=" * 70)
    print("Border Collie Optimization (BCO) Algorithm")
    print("=" * 70)
    print(f"Function: {fname}")
    print(f"Population size: {n}")
    print(f"Maximum iterations: {gen}")
    print(f"Dimensions: {L}")
    print(f"Search space bounds: [{lb}, {ub}]")
    print("=" * 70)
    
    # Initialize the population and acceleration
    initP, acc = generate(n, L, ub, lb)
    
    # Velocity of each individual (initialize to zero)
    Vt = np.zeros((n, L))
    
    # Time of each individual (random between 0 and 1)
    t = np.random.rand(n)
    
    # Best fitness value (initialize to infinity for minimization)
    fopt = np.inf
    
    # Fitness values array
    fit = np.zeros(n)
    
    # Current population
    pop = initP.copy()
    
    # Counter variable for iterations required for Eyeing mechanism
    k = 0  # Start at 0 (MATLAB uses 1, but logic works with 0)
    
    # Store best fitness history
    fopt_history = np.zeros(gen)
    best_position = None
    
    # Store fitness values for analysis
    best_fitness_per_gen = np.zeros(gen)
    
    # Main optimization loop
    for g in range(gen):
        # Calculate fitness of individuals
        fit, maxf, pos = fitness(pop, n, L, ub, lb, fobj)
        
        # Initialize eye flag
        eye = 0
        
        # Update best fitness found so far
        if fopt > maxf:
            fopt = maxf
            best_position = pop[pos, :].copy()
        
        # Store best fitness for this generation
        fopt_history[g] = fopt
        best_fitness_per_gen[g] = maxf
        
        # Check for eyeing mechanism trigger
        # If fitness hasn't improved for 5 consecutive iterations
        if g > 0:
            if fopt_history[g] >= fopt_history[g - 1]:
                k = k + 1
                if k >= 5:
                    eye = 1
                    k = 0
            else:
                # Fitness improved, reset counter
                k = 0
        
        # Sorting the dogs and sheep (herding based on fitness)
        pop, Vt, fit, acc, t = herding(pop, Vt, fit, n, L, acc, t)
        
        # Updating velocity, acceleration, and time of the population
        Vt, acc, t, r1, l1, tempg, temps = update_velocity_time_acceleration(
            Vt, n, L, acc, t, pop, fit, eye
        )
        
        # Updating positions of population
        pop = update(pop, Vt, t, acc, n, L, eye)
        
        # Checking the range of the population is maintained
        pop, acc, t, Vt = check(pop, n, L, ub, lb, acc, Vt, t)
        
        # Print progress
        if verbose and (g % 20 == 0 or g == gen - 1):
            print(f"Generation {g+1:4d}/{gen}: Best fitness = {fopt:.8e}, Current best = {maxf:.8e}")
    
    print("=" * 70)
    print(f"Optimization completed!")
    print(f"Best fitness found: {fopt:.8e}")
    print(f"Best position (first 5 dimensions): {best_position[:5]}")
    print("=" * 70)
    
    return fopt, best_position, fopt_history


def border_collie_optimization_vectorized(fname='func1', n=30, gen=200, verbose=True):
    """
    Vectorized version of BCO algorithm for better performance
    
    Parameters are same as border_collie_optimization()
    """
    
    # Retrieve function details
    fobj, lb, ub, L = func_details(fname)
    
    if verbose:
        print("=" * 70)
        print("Border Collie Optimization (BCO) Algorithm - Vectorized Version")
        print("=" * 70)
        print(f"Function: {fname}")
        print(f"Population size: {n}")
        print(f"Maximum iterations: {gen}")
        print(f"Dimensions: {L}")
        print("=" * 70)
    
    # Initialize
    initP, _ = generate(n, L, ub, lb)
    acc = np.random.rand(n, L)
    Vt = np.zeros((n, L))
    t = np.random.rand(n)
    pop = initP.copy()
    
    fopt = np.inf
    fopt_history = np.zeros(gen)
    best_position = None
    k = 0
    
    for g in range(gen):
        # Calculate fitness
        fit = np.array([fobj(pop[i, :]) for i in range(n)])
        maxf = np.min(fit)
        pos = np.argmin(fit)
        
        eye = 0
        
        if fopt > maxf:
            fopt = maxf
            best_position = pop[pos, :].copy()
        
        fopt_history[g] = fopt
        
        # Eyeing trigger
        if g > 0:
            if fopt_history[g] >= fopt_history[g - 1]:
                k += 1
                if k >= 5:
                    eye = 1
                    k = 0
            else:
                k = 0
        
        # Herding (sort by fitness)
        sorted_idx = np.argsort(fit)
        pop = pop[sorted_idx, :]
        Vt = Vt[sorted_idx, :]
        fit = fit[sorted_idx]
        acc = acc[sorted_idx, :]
        t = t[sorted_idx]
        
        # Update velocities, accelerations, times
        Vt, acc, t, r1, l1, tempg, temps = update_velocity_time_acceleration(
            Vt, n, L, acc, t, pop, fit, eye
        )
        
        # Update positions
        for i in range(n):
            t_i = t[i]
            if i < 3:  # Dogs
                pop[i, :] = Vt[i, :] * t_i + 0.5 * acc[i, :] * (t_i ** 2)
            else:  # Sheep
                if eye == 1:
                    pop[i, :] = Vt[i, :] * t_i - 0.5 * acc[i, :] * (t_i ** 2)
                else:
                    pop[i, :] = Vt[i, :] * t_i + 0.5 * acc[i, :] * (t_i ** 2)
        
        # Check bounds
        if isinstance(ub, (int, float)):
            pop = np.clip(pop, lb, ub)
        else:
            for j in range(L):
                pop[:, j] = np.clip(pop[:, j], lb[j], ub[j])
        
        if verbose and (g % 20 == 0 or g == gen - 1):
            print(f"Generation {g+1:4d}/{gen}: Best fitness = {fopt:.8e}")
    
    if verbose:
        print("=" * 70)
        print(f"Optimization completed! Best fitness: {fopt:.8e}")
        print("=" * 70)
    
    return fopt, best_position, fopt_history


def run_multiple_runs(fname='func1', n=30, gen=200, runs=30, verbose=True):
    """
    Run BCO algorithm multiple times for statistical analysis
    
    Parameters:
    fname : str
        Function name
    n : int
        Population size
    gen : int
        Maximum iterations per run
    runs : int
        Number of independent runs
    verbose : bool
        If True, print progress
    
    Returns:
    results : dict
        Dictionary containing best_fitness, mean, std, min, max for all runs
    all_best_fitness : numpy.ndarray
        Best fitness from each run
    all_positions : list
        Best positions from each run
    all_history : numpy.ndarray
        Fitness history for all runs
    """
    
    all_best_fitness = np.zeros(runs)
    all_positions = []
    all_history = np.zeros((runs, gen))
    
    print("=" * 70)
    print(f"Running BCO on {fname} for {runs} independent runs")
    print("=" * 70)
    
    for run in range(runs):
        if verbose:
            print(f"\nRun {run + 1}/{runs}")
        
        best_fitness, best_position, history = border_collie_optimization(
            fname=fname, n=n, gen=gen, verbose=False
        )
        
        all_best_fitness[run] = best_fitness
        all_positions.append(best_position)
        all_history[run, :] = history
    
    # Calculate statistics
    results = {
        'function': fname,
        'runs': runs,
        'population_size': n,
        'generations': gen,
        'best_fitness': np.min(all_best_fitness),
        'worst_fitness': np.max(all_best_fitness),
        'mean_fitness': np.mean(all_best_fitness),
        'median_fitness': np.median(all_best_fitness),
        'std_fitness': np.std(all_best_fitness),
        'all_best_fitness': all_best_fitness,
        'all_positions': all_positions,
        'all_history': all_history
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("Statistical Summary")
        print("=" * 70)
        print(f"Best fitness: {results['best_fitness']:.8e}")
        print(f"Worst fitness: {results['worst_fitness']:.8e}")
        print(f"Mean fitness: {results['mean_fitness']:.8e}")
        print(f"Median fitness: {results['median_fitness']:.8e}")
        print(f"Std deviation: {results['std_fitness']:.8e}")
        print("=" * 70)
    
    return results, all_best_fitness, all_positions, all_history


# TEST CODE 

if __name__ == "__main__":
    print("=" * 70)
    print("Border Collie Optimization (BCO) Algorithm - Main Program")
    print("=" * 70)
    
    # Test with different functions
    
    # Test 1: Sphere function (func1)
    print("\n" + "=" * 70)
    print("TEST 1: Sphere Function (func1)")
    print("=" * 70)
    
    fopt1, best_pos1, history1 = border_collie_optimization(
        fname='func1', 
        n=30, 
        gen=100, 
        verbose=True
    )
    
    # Test 2: Rosenbrock function (func5)
    print("\n" + "=" * 70)
    print("TEST 2: Rosenbrock Function (func5)")
    print("=" * 70)
    
    fopt2, best_pos2, history2 = border_collie_optimization(
        fname='func5', 
        n=30, 
        gen=100, 
        verbose=True
    )
    
    # Test 3: Rastrigin function (func9)
    print("\n" + "=" * 70)
    print("TEST 3: Rastrigin-like Function (func9)")
    print("=" * 70)
    
    fopt3, best_pos3, history3 = border_collie_optimization(
        fname='func9', 
        n=30, 
        gen=100, 
        verbose=True
    )
    
    # Test 4: Goldstein-Price function (func14 - 2D)
    print("\n" + "=" * 70)
    print("TEST 4: Goldstein-Price Function (func14 - 2D)")
    print("=" * 70)
    
    fopt4, best_pos4, history4 = border_collie_optimization(
        fname='func14', 
        n=20, 
        gen=50, 
        verbose=True
    )
    
    # Test 5: Multiple runs for statistical analysis
    print("\n" + "=" * 70)
    print("TEST 5: Statistical Analysis with Multiple Runs")
    print("=" * 70)
    
    results, all_fitness, all_pos, all_hist = run_multiple_runs(
        fname='func1', 
        n=30, 
        gen=50, 
        runs=10, 
        verbose=True
    )
    
    # Optional: Plot convergence (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # Plot convergence for single run
        plt.subplot(2, 2, 1)
        plt.semilogy(history1, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.title('Convergence - Sphere Function (func1)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.semilogy(history2, 'r-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.title('Convergence - Rosenbrock Function (func5)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.semilogy(history3, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.title('Convergence - Rastrigin-like Function (func9)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Plot all runs
        for i in range(min(5, all_hist.shape[0])):
            plt.semilogy(all_hist[i, :], alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.title(f'Multiple Runs on func1 (n={all_hist.shape[0]} runs)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bco_convergence.png', dpi=150)
        plt.show()
        
        print("\nConvergence plot saved as 'bco_convergence.png'")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    
    print("\n" + "=" * 70)
    print("BCO Algorithm Testing Completed!")
    print("=" * 70)
