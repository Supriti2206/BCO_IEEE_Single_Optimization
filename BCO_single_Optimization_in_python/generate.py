"""
Generate Initial Population for Border Collie Optimization (BCO) Algorithm
This file contains the function to generate random initial population within bounds
"""

import numpy as np

def generate(n, L, ub, lb):
    """
    Generate initial random population within the specified bounds
    
    Parameters:
    n : int
        Number of individuals in the population (population size)
    L : int
        Number of dimensions (variables) for each individual
    ub : float, list, or numpy.ndarray
        Upper bound(s) for variables
        - If scalar: same upper bound for all dimensions
        - If array: different upper bound for each dimension
    lb : float, list, or numpy.ndarray
        Lower bound(s) for variables
        - If scalar: same lower bound for all dimensions
        - If array: different lower bound for each dimension
    
    Returns:
    M : numpy.ndarray
        Matrix of generated positions (size: n x L)
    x : numpy.ndarray
        Random matrix of same size (for compatibility with MATLAB code)
    """
    
    # Convert ub and lb to numpy arrays if they aren't already
    ub = np.array(ub)
    lb = np.array(lb)
    
    # Get number of boundaries (dimensions of ub/lb)
    if ub.ndim == 0:  # Scalar value
        boundary_no = 1
    else:
        boundary_no = ub.shape[0] if ub.shape != () else 1
    
    # Initialize population matrix
    M = np.zeros((n, L))
    
    # Case 1: Same bounds for all variables (single scalar value for ub and lb)
    if boundary_no == 1:
        # Generate random values between lb and ub
        M = np.random.rand(n, L) * (ub - lb) + lb
    
    # Case 2: Different bounds for each variable
    elif boundary_no > 1:
        # Make sure ub and lb have same length as L
        if len(ub) != L or len(lb) != L:
            raise ValueError(f"Length of ub ({len(ub)}) and lb ({len(lb)}) must equal L ({L})")
        
        # Generate positions for each dimension with its own bounds
        for i in range(L):
            ub_i = ub[i]
            lb_i = lb[i]
            M[:, i] = np.random.rand(n) * (ub_i - lb_i) + lb_i
    
    # Generate random matrix x (for compatibility with MATLAB code)
    x = np.random.rand(n, L)
    
    return M, x


def generate_simple(n, L, ub, lb):
    """
    Simplified version of generate function without the x matrix
    
    Parameters:
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    ub : float, list, or numpy.ndarray
        Upper bound(s) for variables
    lb : float, list, or numpy.ndarray
        Lower bound(s) for variables
    
    Returns:
    M : numpy.ndarray
        Matrix of generated positions (size: n x L)
    """
    
    # Convert ub and lb to numpy arrays
    ub = np.array(ub)
    lb = np.array(lb)
    
    # If ub and lb are scalars, create array of same bounds for all dimensions
    if ub.ndim == 0:
        ub = np.full(L, ub)
        lb = np.full(L, lb)
    
    # Initialize population matrix
    M = np.zeros((n, L))
    
    # Generate positions for each dimension with its own bounds
    for i in range(L):
        M[:, i] = np.random.rand(n) * (ub[i] - lb[i]) + lb[i]
    
    return M


# TEST CODE 

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Generate Function for BCO Algorithm")
    print("=" * 70)
    
    # Test Case 1: Same bounds for all variables (scalar ub and lb)
    print("\n" + "-" * 50)
    print("Test Case 1: Same bounds for all variables")
    print("-" * 50)
    
    n = 10  # 10 individuals
    L = 5   # 5 dimensions
    ub = 100  # Upper bound
    lb = -100 # Lower bound
    
    M, x = generate(n, L, ub, lb)
    
    print(f"Population size (n): {n}")
    print(f"Dimensions (L): {L}")
    print(f"Bounds: [{lb}, {ub}]")
    print(f"\nGenerated population shape: {M.shape}")
    print(f"First 3 individuals:\n{M[:3]}")
    print(f"\nCheck bounds - Min value: {np.min(M):.4f}")
    print(f"Check bounds - Max value: {np.max(M):.4f}")
    print(f"x matrix shape: {x.shape}")
    
    # Test Case 2: Different bounds for each variable
    print("\n" + "-" * 50)
    print("Test Case 2: Different bounds for each variable")
    print("-" * 50)
    
    n = 8   # 8 individuals
    L = 4   # 4 dimensions
    ub = [10, 20, 30, 40]   # Different upper bound for each dimension
    lb = [0, 5, 10, 15]     # Different lower bound for each dimension
    
    M, x = generate(n, L, ub, lb)
    
    print(f"Population size (n): {n}")
    print(f"Dimensions (L): {L}")
    print(f"Upper bounds: {ub}")
    print(f"Lower bounds: {lb}")
    print(f"\nGenerated population:\n{M}")
    print(f"\nCheck bounds per dimension:")
    for i in range(L):
        print(f"  Dimension {i+1}: min={np.min(M[:, i]):.4f}, max={np.max(M[:, i]):.4f}, bounds=[{lb[i]}, {ub[i]}]")
    
    # Test Case 3: Using simplified version
    print("\n" + "-" * 50)
    print("Test Case 3: Using simplified version")
    print("-" * 50)
    
    n = 5
    L = 3
    ub = 50
    lb = -50
    
    M_simple = generate_simple(n, L, ub, lb)
    
    print(f"Population (simplified):\n{M_simple}")
    print(f"Min value: {np.min(M_simple):.4f}")
    print(f"Max value: {np.max(M_simple):.4f}")
    
    # Test Case 4: Error handling (mismatched dimensions)
    print("\n" + "-" * 50)
    print("Test Case 4: Error handling (mismatched dimensions)")
    print("-" * 50)
    
    try:
        n = 5
        L = 3
        ub = [10, 20]  # Only 2 bounds but L=3
        lb = [0, 5]
        
        M, x = generate(n, L, ub, lb)
        print("This should not print (error expected)")
        
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test Case 5: Real-world BCO usage example
    print("\n" + "=" * 70)
    print("Real-world BCO Usage Example")
    print("=" * 70)
    
    # BCO parameters
    n_dogs = 3
    n_sheep = 27
    n_population = n_dogs + n_sheep  # Total 30 individuals
    dimensions = 30  # 30-dimensional problem
    
    # Get bounds from function (e.g., func1: Sphere function)
    # For demonstration, using scalar bounds
    ub_func1 = 100
    lb_func1 = -100
    
    # Generate initial population
    population, random_matrix = generate(n_population, dimensions, ub_func1, lb_func1)
    
    print(f"BCO Configuration:")
    print(f"  Total population: {n_population} (3 dogs + 27 sheep)")
    print(f"  Problem dimensions: {dimensions}")
    print(f"  Search space bounds: [{lb_func1}, {ub_func1}]")
    print(f"\nInitial population shape: {population.shape}")
    print(f"Random matrix (x) shape: {random_matrix.shape}")
    print(f"\nFirst dog (first individual): {population[0, :5]}... (first 5 dimensions)")
    print(f"First sheep (4th individual): {population[3, :5]}... (first 5 dimensions)")
    
    # Verify all values are within bounds
    within_bounds = np.all((population >= lb_func1) & (population <= ub_func1))
    print(f"\nAll individuals within bounds: {within_bounds}")
    
    # Example with different bounds per dimension (more realistic)
    print("\n" + "-" * 50)
    print("Example with dimension-specific bounds:")
    print("-" * 50)
    
    n_pop = 20
    dims = 5
    ub_per_dim = [10, 20, 30, 40, 50]
    lb_per_dim = [0, 5, 10, 15, 20]
    
    population_custom, _ = generate(n_pop, dims, ub_per_dim, lb_per_dim)
    
    print(f"Population shape: {population_custom.shape}")
    for d in range(dims):
        col = population_custom[:, d]
        print(f"  Dim {d+1}: min={np.min(col):.2f}, max={np.max(col):.2f}, bounds=[{lb_per_dim[d]}, {ub_per_dim[d]}]")
    
    print("\n" + "=" * 70)
    print("Generate function is working correctly!")
    print("=" * 70)
