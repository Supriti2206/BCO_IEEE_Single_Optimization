"""
Check Function for Border Collie Optimization (BCO) Algorithm
This file contains the function to validate and correct population positions,
accelerations, velocities, and times that are out of bounds or invalid
"""

import numpy as np

def check(pop, n, L, ub, lb, acc, Vt, t):
    """
    Check and correct invalid values in population, acceleration, velocity, and time
    If values are out of bounds, NaN, or zero, reinitialize them with random values
    
    Parameters:
    pop : numpy.ndarray
        Population positions matrix of shape (n, L)
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    ub : float or numpy.ndarray
        Upper bound(s) for variables
    lb : float or numpy.ndarray
        Lower bound(s) for variables
    acc : numpy.ndarray
        Acceleration matrix of shape (n, L)
    Vt : numpy.ndarray
        Velocity matrix of shape (n, L)
    t : numpy.ndarray
        Time array of shape (n,)
    
    Returns:
    pop1 : numpy.ndarray
        Corrected population positions
    acc1 : numpy.ndarray
        Corrected acceleration matrix
    t1 : numpy.ndarray
        Corrected time array
    Vt1 : numpy.ndarray
        Corrected velocity matrix
    """
    
    # Convert ub and lb to numpy arrays if they aren't already
    ub = np.array(ub)
    lb = np.array(lb)
    
    # Handle scalar bounds (same bound for all dimensions)
    if ub.ndim == 0:
        ub = np.full(L, ub)
        lb = np.full(L, lb)
    
    # Initialize corrected arrays as copies of originals
    pop1 = pop.copy()
    acc1 = acc.copy()
    t1 = t.copy()
    Vt1 = Vt.copy()
    
    # Check population positions: out of bounds or zero
    for i in range(n):
        for j in range(L):
            if pop[i, j] >= ub[j] or pop[i, j] <= lb[j] or pop[i, j] == 0:
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
    
    # Check acceleration: NaN or zero
    for i in range(n):
        for j in range(L):
            if np.isnan(acc[i, j]) or acc[i, j] == 0:
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
    
    # Check velocity: NaN or zero
    for i in range(n):
        for j in range(L):
            if np.isnan(Vt[i, j]) or Vt[i, j] == 0:
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
    
    # Check time: NaN or zero
    for i in range(n):
        if np.isnan(t1[i]) or t1[i] == 0:
            for j in range(L):
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
            t1[i] = np.random.rand()
    
    return pop1, acc1, t1, Vt1


def check_vectorized(pop, n, L, ub, lb, acc, Vt, t):
    """
    Vectorized version of check function for better performance
    
    Parameters:
    pop : numpy.ndarray
        Population positions matrix of shape (n, L)
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    ub : float or numpy.ndarray
        Upper bound(s) for variables
    lb : float or numpy.ndarray
        Lower bound(s) for variables
    acc : numpy.ndarray
        Acceleration matrix of shape (n, L)
    Vt : numpy.ndarray
        Velocity matrix of shape (n, L)
    t : numpy.ndarray
        Time array of shape (n,)
    
    Returns:
    pop1 : numpy.ndarray
        Corrected population positions
    acc1 : numpy.ndarray
        Corrected acceleration matrix
    t1 : numpy.ndarray
        Corrected time array
    Vt1 : numpy.ndarray
        Corrected velocity matrix
    """
    
    # Convert ub and lb to numpy arrays
    ub = np.array(ub)
    lb = np.array(lb)
    
    # Handle scalar bounds
    if ub.ndim == 0:
        ub = np.full(L, ub)
        lb = np.full(L, lb)
    
    # Initialize corrected arrays
    pop1 = pop.copy()
    acc1 = acc.copy()
    t1 = t.copy()
    Vt1 = Vt.copy()
    
    # Create bounds for broadcasting
    lb_broadcast = lb.reshape(1, -1)
    ub_broadcast = ub.reshape(1, -1)
    
    # 1. Check population positions: out of bounds or zero
    out_of_bounds = (pop1 >= ub_broadcast) | (pop1 <= lb_broadcast) | (pop1 == 0)
    
    # Generate random corrections where needed
    if np.any(out_of_bounds):
        n_corrections = np.sum(out_of_bounds)
        random_positions = np.random.rand(n_corrections) * (ub_broadcast[0, :] - lb_broadcast[0, :]) + lb_broadcast[0, :]
        
        # Apply corrections
        pop1[out_of_bounds] = random_positions
        
        # For each individual that had a correction, update acc and t
        rows_needing_correction = np.unique(np.where(out_of_bounds)[0])
        for row in rows_needing_correction:
            acc1[row, :] = np.random.rand(L)
            t1[row] = np.random.rand()
    
    # 2. Check acceleration: NaN or zero
    acc_invalid = np.isnan(acc1) | (acc1 == 0)
    
    if np.any(acc_invalid):
        rows_needing_correction = np.unique(np.where(acc_invalid)[0])
        for row in rows_needing_correction:
            # Reinitialize position for that row
            pop1[row, :] = np.random.rand(L) * (ub - lb) + lb
            acc1[row, :] = np.random.rand(L)
            t1[row] = np.random.rand()
    
    # 3. Check velocity: NaN or zero
    Vt_invalid = np.isnan(Vt1) | (Vt1 == 0)
    
    if np.any(Vt_invalid):
        rows_needing_correction = np.unique(np.where(Vt_invalid)[0])
        for row in rows_needing_correction:
            pop1[row, :] = np.random.rand(L) * (ub - lb) + lb
            acc1[row, :] = np.random.rand(L)
            t1[row] = np.random.rand()
    
    # 4. Check time: NaN or zero
    t_invalid = np.isnan(t1) | (t1 == 0)
    
    if np.any(t_invalid):
        rows_needing_correction = np.where(t_invalid)[0]
        for row in rows_needing_correction:
            pop1[row, :] = np.random.rand(L) * (ub - lb) + lb
            acc1[row, :] = np.random.rand(L)
            t1[row] = np.random.rand()
    
    return pop1, acc1, t1, Vt1


def check_with_logging(pop, n, L, ub, lb, acc, Vt, t, verbose=False):
    """
    Check function with logging to track corrections made
    
    Parameters:
    pop, n, L, ub, lb, acc, Vt, t : same as check()
    verbose : bool
        If True, print information about corrections made
    
    Returns:
    pop1, acc1, t1, Vt1 : corrected arrays
    corrections : dict
        Dictionary containing counts of corrections made
    """
    
    # Convert ub and lb to numpy arrays
    ub = np.array(ub)
    lb = np.array(lb)
    
    if ub.ndim == 0:
        ub = np.full(L, ub)
        lb = np.full(L, lb)
    
    # Initialize
    pop1 = pop.copy()
    acc1 = acc.copy()
    t1 = t.copy()
    Vt1 = Vt.copy()
    
    corrections = {
        'position_out_of_bounds': 0,
        'position_zero': 0,
        'acceleration_nan': 0,
        'acceleration_zero': 0,
        'velocity_nan': 0,
        'velocity_zero': 0,
        'time_nan': 0,
        'time_zero': 0,
        'individuals_corrected': set()
    }
    
    # Check population positions
    for i in range(n):
        for j in range(L):
            if pop[i, j] >= ub[j] or pop[i, j] <= lb[j]:
                corrections['position_out_of_bounds'] += 1
                corrections['individuals_corrected'].add(i)
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
            elif pop[i, j] == 0:
                corrections['position_zero'] += 1
                corrections['individuals_corrected'].add(i)
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
    
    # Check acceleration
    for i in range(n):
        for j in range(L):
            if np.isnan(acc[i, j]):
                corrections['acceleration_nan'] += 1
                corrections['individuals_corrected'].add(i)
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
            elif acc[i, j] == 0:
                corrections['acceleration_zero'] += 1
                corrections['individuals_corrected'].add(i)
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
    
    # Check velocity
    for i in range(n):
        for j in range(L):
            if np.isnan(Vt[i, j]):
                corrections['velocity_nan'] += 1
                corrections['individuals_corrected'].add(i)
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
            elif Vt[i, j] == 0:
                corrections['velocity_zero'] += 1
                corrections['individuals_corrected'].add(i)
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
                t1[i] = np.random.rand()
    
    # Check time
    for i in range(n):
        if np.isnan(t1[i]):
            corrections['time_nan'] += 1
            corrections['individuals_corrected'].add(i)
            for j in range(L):
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
            t1[i] = np.random.rand()
        elif t1[i] == 0:
            corrections['time_zero'] += 1
            corrections['individuals_corrected'].add(i)
            for j in range(L):
                pop1[i, j] = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                acc1[i, j] = np.random.rand()
            t1[i] = np.random.rand()
    
    corrections['individuals_corrected'] = len(corrections['individuals_corrected'])
    
    if verbose:
        print("\n" + "=" * 50)
        print("CHECK FUNCTION CORRECTIONS REPORT")
        print("=" * 50)
        print(f"Position out of bounds: {corrections['position_out_of_bounds']}")
        print(f"Position zero: {corrections['position_zero']}")
        print(f"Acceleration NaN: {corrections['acceleration_nan']}")
        print(f"Acceleration zero: {corrections['acceleration_zero']}")
        print(f"Velocity NaN: {corrections['velocity_nan']}")
        print(f"Velocity zero: {corrections['velocity_zero']}")
        print(f"Time NaN: {corrections['time_nan']}")
        print(f"Time zero: {corrections['time_zero']}")
        print(f"Total individuals corrected: {corrections['individuals_corrected']}")
        print("=" * 50)
    
    return pop1, acc1, t1, Vt1, corrections


# TEST CODE 

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Check Function for BCO Algorithm")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    n = 10      # 10 individuals
    L = 5       # 5 dimensions
    ub = 10     # Upper bound
    lb = -10    # Lower bound
    
    # Create test data with intentional invalid values
    pop = np.random.randn(n, L) * 15  # Some values will be out of bounds (outside -10 to 10)
    acc = np.random.randn(n, L) * 0.5
    Vt = np.random.randn(n, L) * 0.5
    t = np.random.rand(n) * 2
    
    # Introduce specific invalid values for testing
    pop[0, 0] = 100      # Out of bounds (too high)
    pop[1, 1] = -100     # Out of bounds (too low)
    pop[2, 2] = 0        # Zero value
    acc[3, 0] = np.nan   # NaN acceleration
    acc[4, 1] = 0        # Zero acceleration
    Vt[5, 2] = np.nan    # NaN velocity
    Vt[6, 3] = 0         # Zero velocity
    t[7] = np.nan        # NaN time
    t[8] = 0             # Zero time
    
    print("Original Data with Invalid Values:")
    print("-" * 50)
    print(f"Population - first few values:")
    for i in range(min(5, n)):
        print(f"  Individual {i}: {pop[i, :3]}...")
    print(f"\nAcceleration - first few:")
    for i in range(min(5, n)):
        print(f"  Individual {i}: {acc[i, :3]}...")
    print(f"\nVelocity - first few:")
    for i in range(min(5, n)):
        print(f"  Individual {i}: {Vt[i, :3]}...")
    print(f"\nTime: {t}")
    
    # Test Case 1: Basic check function
    print("\n" + "=" * 70)
    print("Test Case 1: Basic Check Function")
    print("=" * 70)
    
    pop1, acc1, t1, Vt1 = check(pop, n, L, ub, lb, acc, Vt, t)
    
    print("\nCorrected Data:")
    print(f"Population - first few values:")
    for i in range(min(5, n)):
        print(f"  Individual {i}: {pop1[i, :3]}...")
    
    # Verify corrections
    print("\nVerification:")
    print(f"Individual 0, dim 0 was 100, now: {pop1[0, 0]:.4f} (should be between {lb} and {ub})")
    print(f"Individual 1, dim 1 was -100, now: {pop1[1, 1]:.4f} (should be between {lb} and {ub})")
    print(f"Individual 2, dim 2 was 0, now: {pop1[2, 2]:.4f} (should not be 0)")
    print(f"Individual 3, dim 0 acceleration was NaN, now: {acc1[3, 0]:.4f}")
    print(f"Individual 4, dim 1 acceleration was 0, now: {acc1[4, 1]:.4f}")
    print(f"Individual 5, dim 2 velocity was NaN, now: {Vt1[5, 2]:.4f}")
    print(f"Individual 6, dim 3 velocity was 0, now: {Vt1[6, 3]:.4f}")
    print(f"Individual 7 time was NaN, now: {t1[7]:.4f}")
    print(f"Individual 8 time was 0, now: {t1[8]:.4f}")
    
    # Test Case 2: Check with logging
    print("\n" + "=" * 70)
    print("Test Case 2: Check Function with Logging")
    print("=" * 70)
    
    # Create fresh test data
    pop_test = np.random.randn(n, L) * 15
    acc_test = np.random.randn(n, L) * 0.5
    Vt_test = np.random.randn(n, L) * 0.5
    t_test = np.random.rand(n) * 2
    
    # Introduce invalid values
    pop_test[0, :] = 100
    acc_test[1, 0] = np.nan
    Vt_test[2, 1] = 0
    t_test[3] = np.nan
    
    pop2, acc2, t2, Vt2, corrections = check_with_logging(
        pop_test, n, L, ub, lb, acc_test, Vt_test, t_test, verbose=True
    )
    
    # Test Case 3: Different bounds per dimension
    print("\n" + "=" * 70)
    print("Test Case 3: Different Bounds per Dimension")
    print("=" * 70)
    
    n2 = 8
    L2 = 4
    ub_per_dim = [10, 20, 30, 40]
    lb_per_dim = [0, 5, 10, 15]
    
    pop_diff = np.random.randn(n2, L2) * 50
    acc_diff = np.random.randn(n2, L2) * 0.5
    Vt_diff = np.random.randn(n2, L2) * 0.5
    t_diff = np.random.rand(n2) * 2
    
    # Introduce invalid values
    pop_diff[0, 0] = -5    # Below lb[0]=0
    pop_diff[1, 1] = 30    # Above ub[1]=20
    pop_diff[2, 2] = 0     # Zero
    
    print(f"Bounds per dimension:")
    for d in range(L2):
        print(f"  Dimension {d}: [{lb_per_dim[d]}, {ub_per_dim[d]}]")
    
    pop3, acc3, t3, Vt3 = check(pop_diff, n2, L2, ub_per_dim, lb_per_dim, acc_diff, Vt_diff, t_diff)
    
    print(f"\nCorrected values:")
    print(f"Individual 0, dim 0 was -5, now: {pop3[0, 0]:.4f} (should be >= {lb_per_dim[0]})")
    print(f"Individual 1, dim 1 was 30, now: {pop3[1, 1]:.4f} (should be <= {ub_per_dim[1]})")
    print(f"Individual 2, dim 2 was 0, now: {pop3[2, 2]:.4f} (should not be 0)")
    
    # Test Case 4: Real BCO scenario
    print("\n" + "=" * 70)
    print("Test Case 4: Real BCO Scenario Simulation")
    print("=" * 70)
    
    n_total = 30   # 3 dogs + 27 sheep
    dims = 10
    ub_bco = 100
    lb_bco = -100
    
    # Simulate BCO iteration that might produce invalid values
    pop_bco = np.random.randn(n_total, dims) * 150  # Many values out of bounds
    acc_bco = np.random.randn(n_total, dims) * 2
    Vt_bco = np.random.randn(n_total, dims) * 2
    t_bco = np.random.rand(n_total) * 3
    
    # Introduce zeros
    acc_bco[5, :] = 0
    Vt_bco[10, :] = 0
    t_bco[15] = 0
    
    print(f"Before check - Invalid values summary:")
    out_of_bounds_count = np.sum((pop_bco > ub_bco) | (pop_bco < lb_bco))
    print(f"  Positions out of bounds: {out_of_bounds_count}")
    print(f"  Zero accelerations: {np.sum(acc_bco == 0)}")
    print(f"  Zero velocities: {np.sum(Vt_bco == 0)}")
    print(f"  Zero times: {np.sum(t_bco == 0)}")
    
    # Apply check
    pop_bco_corrected, acc_bco_corrected, t_bco_corrected, Vt_bco_corrected = check(
        pop_bco, n_total, dims, ub_bco, lb_bco, acc_bco, Vt_bco, t_bco
    )
    
    print(f"\nAfter check - Invalid values summary:")
    out_of_bounds_corrected = np.sum((pop_bco_corrected > ub_bco) | (pop_bco_corrected < lb_bco))
    print(f"  Positions out of bounds: {out_of_bounds_corrected}")
    print(f"  Zero accelerations: {np.sum(acc_bco_corrected == 0)}")
    print(f"  Zero velocities: {np.sum(Vt_bco_corrected == 0)}")
    print(f"  Zero times: {np.sum(t_bco_corrected == 0)}")
    
    print(f"\nAll values valid: {out_of_bounds_corrected == 0}")
    
    # Test Case 5: Performance comparison
    print("\n" + "=" * 70)
    print("Performance Comparison (500 runs)")
    print("=" * 70)
    
    import time
    
    # Create large test data
    n_large = 100
    L_large = 50
    pop_large = np.random.randn(n_large, L_large) * 100
    acc_large = np.random.randn(n_large, L_large)
    Vt_large = np.random.randn(n_large, L_large)
    t_large = np.random.rand(n_large)
    
    # Regular version
    start_time = time.time()
    for _ in range(500):
        pop_reg, acc_reg, t_reg, Vt_reg = check(pop_large, n_large, L_large, 10, -10, acc_large, Vt_large, t_large)
    reg_time = time.time() - start_time
    
    # Vectorized version
    start_time = time.time()
    for _ in range(500):
        pop_vec, acc_vec, t_vec, Vt_vec = check_vectorized(pop_large, n_large, L_large, 10, -10, acc_large, Vt_large, t_large)
    vec_time = time.time() - start_time
    
    print(f"Regular version time: {reg_time:.4f} seconds")
    print(f"Vectorized version time: {vec_time:.4f} seconds")
    print(f"Vectorized is {reg_time/vec_time:.2f}x faster")
    
    print("\n" + "=" * 70)
    print("Check function is working correctly!")
    print("=" * 70)
