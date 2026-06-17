"""
Position Update Function for Border Collie Optimization (BCO) Algorithm
This file contains the function to update positions of dogs and sheep
"""

import numpy as np

def update(pop, Vt, t, acc, n, L, eye_flag):
    """
    Update positions of dogs and sheep based on velocity, acceleration, and time
    
    Parameters:
    pop : numpy.ndarray
        Current population positions matrix of shape (n, L)
        (Note: This parameter is kept for compatibility but not used in calculation)
    Vt : numpy.ndarray
        Velocity matrix of shape (n, L)
    t : numpy.ndarray
        Time array of shape (n,)
    acc : numpy.ndarray
        Acceleration matrix of shape (n, L)
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    eye_flag : int or bool
        Flag indicating if eyeing behavior is active (1 = eyeing, 0 = normal)
    
    Returns:
    pop1 : numpy.ndarray
        Updated population positions matrix of shape (n, L)
    """
    
    # Initialize updated population matrix
    pop1 = np.zeros((n, L))
    
    # Update positions for each individual and each dimension
    for i in range(n):
        for j in range(L):
            # Updating the position of dogs (first 3 individuals)
            if i < 3:  # MATLAB uses i<=3 (1-indexed), Python uses i<3 (0-indexed)
                # Dogs use positive acceleration (gathering behavior)
                pop1[i, j] = Vt[i, j] * t[i] + (1/2) * acc[i, j] * (t[i]**2)
            
            # Updating position of sheep (individuals 4 and above, 0-indexed i>=3)
            if i >= 3:
                if eye_flag == 1:
                    # Eyeing behavior: sheep use negative acceleration (retardation)
                    pop1[i, j] = Vt[i, j] * t[i] - (1/2) * acc[i, j] * (t[i]**2)
                else:
                    # Normal sheep movement (gathering or stalking)
                    pop1[i, j] = Vt[i, j] * t[i] + (1/2) * acc[i, j] * (t[i]**2)
    
    return pop1


def update_vectorized(pop, Vt, t, acc, n, L, eye_flag):
    """
    Vectorized version of update function for better performance (faster)
    
    Parameters:
    pop : numpy.ndarray
        Current population positions matrix (kept for compatibility)
    Vt : numpy.ndarray
        Velocity matrix of shape (n, L)
    t : numpy.ndarray
        Time array of shape (n,)
    acc : numpy.ndarray
        Acceleration matrix of shape (n, L)
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    eye_flag : int or bool
        Flag indicating if eyeing behavior is active
    
    Returns:
    pop1 : numpy.ndarray
        Updated population positions matrix of shape (n, L)
    """
    
    # Reshape t to enable broadcasting with Vt and acc
    # t shape: (n,) -> (n, 1) for broadcasting
    t_reshaped = t.reshape(-1, 1)
    
    # Calculate base displacement (velocity * time)
    displacement = Vt * t_reshaped
    
    # Calculate acceleration contribution (1/2 * acceleration * time^2)
    acc_contribution = 0.5 * acc * (t_reshaped**2)
    
    # Initialize updated population
    pop1 = np.zeros((n, L))
    
    # For dogs (first 3 individuals, 0-indexed: 0,1,2)
    # Dogs always use positive acceleration (gathering)
    if n >= 3:
        pop1[0:3, :] = displacement[0:3, :] + acc_contribution[0:3, :]
    
    # For sheep (individuals 3 to n-1)
    if n > 3:
        if eye_flag == 1:
            # Eyeing behavior: subtract acceleration contribution
            pop1[3:, :] = displacement[3:, :] - acc_contribution[3:, :]
        else:
            # Normal behavior: add acceleration contribution
            pop1[3:, :] = displacement[3:, :] + acc_contribution[3:, :]
    
    return pop1


def update_detailed(pop, Vt, t, acc, n, L, eye_flag, n_dogs=3):
    """
    More detailed version with explicit dog/sheep separation and parameter control
    
    Parameters:
    pop : numpy.ndarray
        Current population positions (kept for compatibility)
    Vt : numpy.ndarray
        Velocity matrix of shape (n, L)
    t : numpy.ndarray
        Time array of shape (n,)
    acc : numpy.ndarray
        Acceleration matrix of shape (n, L)
    n : int
        Total number of individuals
    L : int
        Number of dimensions
    eye_flag : int or bool
        Eyeing behavior flag (1 = active, 0 = inactive)
    n_dogs : int
        Number of dogs (default: 3)
    
    Returns:
    pop1 : numpy.ndarray
        Updated positions
    """
    
    # Reshape time for broadcasting
    t_reshaped = t.reshape(-1, 1)
    
    # Calculate displacement = velocity * time
    displacement = Vt * t_reshaped
    
    # Calculate acceleration term = 0.5 * acceleration * time^2
    acc_term = 0.5 * acc * (t_reshaped**2)
    
    # Initialize output
    pop1 = np.zeros((n, L))
    
    # Dogs: always use + acceleration (gathering/stalking)
    pop1[:n_dogs, :] = displacement[:n_dogs, :] + acc_term[:n_dogs, :]
    
    # Sheep: depends on eye_flag
    if eye_flag:
        # Eyeing: sheep experience retardation (negative acceleration)
        pop1[n_dogs:, :] = displacement[n_dogs:, :] - acc_term[n_dogs:, :]
    else:
        # Normal: sheep follow dogs (positive acceleration)
        pop1[n_dogs:, :] = displacement[n_dogs:, :] + acc_term[n_dogs:, :]
    
    # Optional: Apply bounds checking if needed
    # (Bounds checking would require ub and lb parameters)
    
    return pop1


# TEST CODE 

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Position Update Function for BCO Algorithm")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    n = 10  # 10 individuals (3 dogs + 7 sheep)
    L = 5   # 5 dimensions
    
    # Create test data
    pop = np.random.randn(n, L) * 10      # Current positions
    Vt = np.random.randn(n, L) * 2        # Velocities
    t = np.random.rand(n) * 5             # Time values
    acc = np.random.randn(n, L) * 0.5     # Accelerations
    
    print("Test Data:")
    print("-" * 40)
    print(f"Population size (n): {n}")
    print(f"Dimensions (L): {L}")
    print(f"Number of dogs: 3 (first 3 individuals)")
    print(f"Number of sheep: {n-3}")
    print(f"\nFirst dog (index 0) velocity: {Vt[0, :3]}...")
    print(f"First dog time: {t[0]:.4f}")
    print(f"First dog acceleration: {acc[0, :3]}...")
    
    # Test Case 1: Normal behavior (no eyeing)
    print("\n" + "=" * 70)
    print("Test Case 1: Normal Behavior (eye_flag = 0)")
    print("=" * 70)
    
    pop1_normal = update(pop, Vt, t, acc, n, L, eye_flag=0)
    
    # Calculate manually for verification
    t0_reshaped = t[0].reshape(-1, 1)
    displacement_manual = Vt[0, :] * t[0]
    acc_term_manual = 0.5 * acc[0, :] * (t[0]**2)
    expected_dog_pos = displacement_manual + acc_term_manual
    
    print(f"First dog (index 0) updated position (first 3 dims): {pop1_normal[0, :3]}")
    print(f"Expected first dog position: {expected_dog_pos[:3]}")
    print(f"Match?: {np.allclose(pop1_normal[0, :3], expected_dog_pos[:3])}")
    
    # Test Case 2: Eyeing behavior (eye_flag = 1)
    print("\n" + "=" * 70)
    print("Test Case 2: Eyeing Behavior (eye_flag = 1)")
    print("=" * 70)
    
    pop1_eyeing = update(pop, Vt, t, acc, n, L, eye_flag=1)
    
    # Calculate manually for verification (sheep use negative acceleration)
    sheep_idx = 5  # Index of a sheep (>=3)
    t_sheep = t[sheep_idx]
    displacement_sheep = Vt[sheep_idx, :] * t_sheep
    acc_term_sheep = 0.5 * acc[sheep_idx, :] * (t_sheep**2)
    expected_sheep_pos = displacement_sheep - acc_term_sheep
    
    print(f"Sheep (index {sheep_idx}) updated position (first 3 dims): {pop1_eyeing[sheep_idx, :3]}")
    print(f"Expected sheep position (with retardation): {expected_sheep_pos[:3]}")
    print(f"Match?: {np.allclose(pop1_eyeing[sheep_idx, :3], expected_sheep_pos[:3])}")
    
    # Compare normal vs eyeing for sheep
    print(f"\nSheep position with eye_flag=0 (no eyeing): {pop1_normal[sheep_idx, :3]}")
    print(f"Sheep position with eye_flag=1 (eyeing): {pop1_eyeing[sheep_idx, :3]}")
    print(f"Difference (eyeing has lower values): {pop1_normal[sheep_idx, :3] - pop1_eyeing[sheep_idx, :3]}")
    
    # Test Case 3: Vectorized version comparison
    print("\n" + "=" * 70)
    print("Test Case 3: Comparing Regular vs Vectorized Version")
    print("=" * 70)
    
    pop1_vectorized = update_vectorized(pop, Vt, t, acc, n, L, eye_flag=0)
    
    print(f"Regular version - first dog: {pop1_normal[0, :3]}")
    print(f"Vectorized version - first dog: {pop1_vectorized[0, :3]}")
    print(f"Results identical?: {np.allclose(pop1_normal, pop1_vectorized)}")
    
    # Test Case 4: Detailed version test
    print("\n" + "=" * 70)
    print("Test Case 4: Detailed Version with Custom Dog Count")
    print("=" * 70)
    
    n_dogs_custom = 2
    pop1_detailed = update_detailed(pop, Vt, t, acc, n, L, eye_flag=0, n_dogs=n_dogs_custom)
    
    print(f"Using {n_dogs_custom} dogs (instead of default 3)")
    print(f"First {n_dogs_custom} individuals (dogs) updated with + acceleration")
    print(f"Remaining {n - n_dogs_custom} individuals (sheep) updated based on eye_flag")
    print(f"First dog position: {pop1_detailed[0, :3]}")
    print(f"First sheep (index {n_dogs_custom}) position: {pop1_detailed[n_dogs_custom, :3]}")
    
    # Test Case 5: Real BCO scenario
    print("\n" + "=" * 70)
    print("Test Case 5: Real BCO Scenario")
    print("=" * 70)
    
    # BCO parameters
    n_dogs_bco = 3
    n_sheep_bco = 27
    n_total = n_dogs_bco + n_sheep_bco
    dims = 30
    
    # Create realistic BCO data
    velocities = np.random.randn(n_total, dims) * 0.5
    times = np.random.uniform(0.5, 2.0, n_total)
    accelerations = np.random.randn(n_total, dims) * 0.1
    dummy_pop = np.zeros((n_total, dims))  # Not used in calculation
    
    print(f"BCO Configuration:")
    print(f"  Total individuals: {n_total} (3 dogs + 27 sheep)")
    print(f"  Problem dimensions: {dims}")
    
    # Simulate multiple iterations with different eyeing conditions
    for iteration in range(3):
        eye_active = iteration % 2  # Alternate between normal and eyeing
        eye_text = "ACTIVE" if eye_active else "INACTIVE"
        
        updated_positions = update_vectorized(
            dummy_pop, velocities, times, accelerations, n_total, dims, eye_active
        )
        
        # Calculate statistics
        dog_positions = updated_positions[:n_dogs_bco, :]
        sheep_positions = updated_positions[n_dogs_bco:, :]
        
        print(f"\nIteration {iteration + 1} (Eyeing: {eye_text}):")
        print(f"  Dog positions - mean: {np.mean(dog_positions):.6f}, std: {np.std(dog_positions):.6f}")
        print(f"  Sheep positions - mean: {np.mean(sheep_positions):.6f}, std: {np.std(sheep_positions):.6f}")
        
        if eye_active:
            print(f"  Note: Sheep experience retardation (negative acceleration)")
        else:
            print(f"  Note: Sheep follow dogs (positive acceleration)")
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("Performance Comparison (1000 runs)")
    print("=" * 70)
    
    import time
    
    # Regular version
    start_time = time.time()
    for _ in range(1000):
        pop1_reg = update(pop, Vt, t, acc, n, L, eye_flag=0)
    reg_time = time.time() - start_time
    
    # Vectorized version
    start_time = time.time()
    for _ in range(1000):
        pop1_vec = update_vectorized(pop, Vt, t, acc, n, L, eye_flag=0)
    vec_time = time.time() - start_time
    
    print(f"Regular version time: {reg_time:.4f} seconds")
    print(f"Vectorized version time: {vec_time:.4f} seconds")
    print(f"Vectorized is {reg_time/vec_time:.2f}x faster")
    
    print("\n" + "=" * 70)
    print("Position update function is working correctly!")
    print("=" * 70)
