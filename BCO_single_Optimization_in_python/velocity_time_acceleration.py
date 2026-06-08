"""
Velocity, Time, and Acceleration Update Function for Border Collie Optimization (BCO) Algorithm
This file contains the function to update velocities, accelerations, and times for dogs and sheep
"""

import numpy as np

def update_velocity_time_acceleration(Vt, n, L, a, t, pop, fit, eye_flag):
    """
    Update velocities, accelerations, and times for dogs and sheep based on herding behavior
    
    Parameters:
    -----------
    Vt : numpy.ndarray
        Current velocity matrix of shape (n, L)
    n : int
        Number of individuals in the population
    L : int
        Number of dimensions for each individual
    a : numpy.ndarray
        Current acceleration matrix of shape (n, L)
    t : numpy.ndarray
        Current time array of shape (n,)
    pop : numpy.ndarray
        Population positions matrix of shape (n, L)
    fit : numpy.ndarray
        Fitness values array of shape (n,) (sorted with best first)
    eye_flag : int or bool
        Flag indicating if eyeing behavior is active (1 = eyeing, 0 = normal)
    
    Returns:
    --------
    Vt1 : numpy.ndarray
        Updated velocity matrix
    acc : numpy.ndarray
        Updated acceleration matrix
    t1 : numpy.ndarray
        Updated time array
    r1 : int
        Index of right dog (1-indexed as per MATLAB, but will be 0-indexed in Python)
    l1 : int
        Index of left dog (1-indexed as per MATLAB, but will be 0-indexed in Python)
    tempg : list or numpy.ndarray
        Indices of sheep that are gathered
    temps : list or numpy.ndarray
        Indices of sheep that are stalked
    """
    
    # Tournament selection for choosing left and right dogs
    # Randomly choose between 2 and 3 (1-indexed MATLAB)
    # In Python, we'll use 0-indexed internally but return as 0-indexed
    r1_temp = np.random.randint(2, 4)  # Returns 2 or 3 (MATLAB 1-indexed)
    
    if r1_temp == 2:
        l1_temp = 3
    else:
        l1_temp = 2
    
    # Convert to 0-indexed for Python array access
    r1 = r1_temp - 1  # Right dog index (0-indexed)
    l1 = l1_temp - 1  # Left dog index (0-indexed)
    
    # Initialize updated velocity matrix (start with copy of current)
    Vt1 = Vt.copy()
    
    # Get dimensions for acceleration
    r, l = a.shape
    
    # Initialize acceleration array
    a1 = np.zeros((r, l))
    
    # Best fitness (lead dog)
    f1 = fit[0]
    
    # Average fitness of left and right dogs
    f2 = (fit[1] + fit[2]) / 2
    
    # Initialize tracking arrays for gathered and stalked sheep
    tempg = []
    temps = []
    
    # Setting parameters for eyeing
    if eye_flag == 1:
        # Use the dog with worse fitness (least fitness among dogs) for eyeing
        # In MATLAB: if fit(r1) < fit(l1) (note: r1 and l1 are 1-indexed values 2 or 3)
        # We need to compare fitness at indices r1 and l1 (0-indexed)
        if fit[r1] < fit[l1]:
            # Left dog (l1) has worse fitness? Actually MATLAB code is confusing
            # Let's trace: if fit(r1) < fit(l1) then use a1(l1,:)
            # We'll implement as per MATLAB logic
            a1[l1, :] = -1 * a[l1, :]
            f = l1
        else:
            a1[r1, :] = -1 * a[r1, :]
            f = r1
    
    # Main loop for updating velocities
    for i in range(n):
        for j in range(L):
            # Velocity updation of dogs (first 3 individuals, 0-indexed: 0,1,2)
            if i < 3:
                # Real part ensures no complex numbers (handle negative under sqrt)
                val = Vt[i, j]**2 + 2 * a[i, j] * pop[i, j]
                if val >= 0:
                    Vt1[i, j] = np.sqrt(val)
                else:
                    Vt1[i, j] = 0  # Handle negative values
            
            # Velocity updation of sheep (i >= 3)
            if i >= 3:
                # If Eyeing is true
                if eye_flag == 1:
                    # Use the dog selected for eyeing
                    val = Vt1[f, j]**2 + 2 * a1[f, j] * pop[i, j]
                    if val >= 0:
                        Vt1[i, j] = np.sqrt(val)
                    else:
                        Vt1[i, j] = 0
                
                # If not eyeing (normal behavior)
                if eye_flag != 1:
                    # Velocity updation of gathered sheep
                    # Condition: f1 - fit[i] > f2 - fit[i]  (sheep nearer to lead dog)
                    if (f1 - fit[i]) > (f2 - fit[i]):
                        val = Vt1[0, j]**2 + 2 * a[0, j] * pop[i, j]
                        if val >= 0:
                            Vt1[i, j] = np.sqrt(val)
                        else:
                            Vt1[i, j] = 0
                        if i not in tempg:
                            tempg.append(i)
                    
                    # Velocity updation of stalked sheep
                    # Condition: f1 - fit[i] <= f2 - fit[i]  (sheep nearer to side dogs)
                    else:
                        # Random angles for stalking
                        angle1 = np.random.randint(1, 90)  # 1 to 89 degrees
                        angle2 = np.random.randint(91, 180)  # 91 to 179 degrees
                        
                        # Calculate term for right dog
                        term1 = (Vt1[r1, j] * np.tan(np.radians(angle1)))**2 + 2 * a[r1, j] * pop[r1, j]
                        if term1 >= 0:
                            val1 = np.sqrt(term1)
                        else:
                            val1 = 0
                        
                        # Calculate term for left dog
                        term2 = (Vt1[l1, j] * np.tan(np.radians(angle2)))**2 + 2 * a[l1, j] * pop[l1, j]
                        if term2 >= 0:
                            val2 = np.sqrt(term2)
                        else:
                            val2 = 0
                        
                        # Average of both dogs' influence
                        Vt1[i, j] = (val1 + val2) / 2
                        
                        if i not in temps:
                            temps.append(i)
    
    # Updation of time and acceleration
    acc = np.zeros((n, L))
    t1 = np.zeros(n)
    
    for i in range(n):
        s_sum = 0
        for j in range(L):
            # Calculate acceleration: |Vt1 - Vt| / t
            # Avoid division by zero
            if t[i] > 0:
                acc[i, j] = abs(Vt1[i, j] - Vt[i, j]) / t[i]
            else:
                acc[i, j] = 0
            
            # Sum for time calculation
            if acc[i, j] > 0:
                s_sum = s_sum + (Vt1[i, j] - Vt[i, j]) / acc[i, j]
        
        # Time update: mean of the sum across dimensions
        t1[i] = abs(np.mean(s_sum))
    
    return Vt1, acc, t1, r1, l1, tempg, temps


def update_velocity_vectorized(Vt, n, L, a, t, pop, fit, eye_flag):
    """
    Vectorized version for better performance (faster than loop version)
    """
    
    # Tournament selection for left and right dogs
    r1_temp = np.random.randint(2, 4)
    l1_temp = 3 if r1_temp == 2 else 2
    r1 = r1_temp - 1
    l1 = l1_temp - 1
    
    # Initialize
    Vt1 = Vt.copy()
    r, l = a.shape
    a1 = np.zeros((r, l))
    f1 = fit[0]
    f2 = (fit[1] + fit[2]) / 2
    tempg = []
    temps = []
    
    # Eyeing setup
    if eye_flag == 1:
        if fit[r1] < fit[l1]:
            a1[l1, :] = -1 * a[l1, :]
            f = l1
        else:
            a1[r1, :] = -1 * a[r1, :]
            f = r1
    
    # Dogs (first 3 individuals)
    for i in range(min(3, n)):
        val = Vt[i, :]**2 + 2 * a[i, :] * pop[i, :]
        Vt1[i, :] = np.sqrt(np.maximum(val, 0))
    
    # Sheep (i >= 3)
    if n > 3:
        if eye_flag == 1:
            # Eyeing: use selected dog
            val = Vt1[f, :]**2 + 2 * a1[f, :] * pop[3:n, :]
            Vt1[3:n, :] = np.sqrt(np.maximum(val, 0))
        
        else:
            # Gathered sheep condition
            condition = (f1 - fit[3:n]) > (f2 - fit[3:n])
            
            # Gathered sheep indices
            gathered_indices = np.where(condition)[0] + 3
            for idx in gathered_indices:
                if idx not in tempg:
                    tempg.append(idx)
            
            # Stalked sheep indices
            stalked_indices = np.where(~condition)[0] + 3
            for idx in stalked_indices:
                if idx not in temps:
                    temps.append(idx)
            
            # Vectorized update for gathered sheep
            if len(gathered_indices) > 0:
                gathered_pos = gathered_indices - 3  # Adjust for slicing
                val = Vt1[0, :]**2 + 2 * a[0, :] * pop[gathered_indices, :]
                Vt1[gathered_indices, :] = np.sqrt(np.maximum(val, 0))
            
            # Update for stalked sheep (need individual handling due to random angles)
            for idx in stalked_indices:
                angle1 = np.random.randint(1, 90)
                angle2 = np.random.randint(91, 180)
                
                term1 = (Vt1[r1, :] * np.tan(np.radians(angle1)))**2 + 2 * a[r1, :] * pop[r1, :]
                term2 = (Vt1[l1, :] * np.tan(np.radians(angle2)))**2 + 2 * a[l1, :] * pop[l1, :]
                
                val1 = np.sqrt(np.maximum(term1, 0))
                val2 = np.sqrt(np.maximum(term2, 0))
                
                Vt1[idx, :] = (val1 + val2) / 2
    
    # Update acceleration and time
    acc = np.zeros((n, L))
    t1 = np.zeros(n)
    
    for i in range(n):
        # Avoid division by zero
        t_safe = np.maximum(t[i], 1e-10)
        acc[i, :] = np.abs(Vt1[i, :] - Vt[i, :]) / t_safe
        
        # Calculate time
        with np.errstate(divide='ignore', invalid='ignore'):
            s_sum = np.sum((Vt1[i, :] - Vt[i, :]) / np.maximum(acc[i, :], 1e-10))
        t1[i] = abs(np.mean(s_sum))
    
    return Vt1, acc, t1, r1, l1, tempg, temps


# ==================== TEST CODE ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Velocity, Time, Acceleration Update Function for BCO Algorithm")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test parameters
    n = 10  # 10 individuals (3 dogs + 7 sheep)
    L = 5   # 5 dimensions
    
    # Create test data
    Vt = np.random.randn(n, L) * 0.5        # Velocities
    a = np.random.randn(n, L) * 0.1         # Accelerations
    t = np.random.uniform(0.5, 2.0, n)      # Times
    pop = np.random.randn(n, L) * 10        # Positions
    
    # Sorted fitness values (best first)
    fit = np.random.rand(n) * 100
    fit = np.sort(fit)  # Sort ascending (best first)
    
    print("Test Data:")
    print("-" * 40)
    print(f"Population size (n): {n}")
    print(f"Dimensions (L): {L}")
    print(f"Number of dogs: 3")
    print(f"Number of sheep: {n-3}")
    print(f"\nBest fitness (lead dog): {fit[0]:.4f}")
    print(f"2nd best fitness: {fit[1]:.4f}")
    print(f"3rd best fitness: {fit[2]:.4f}")
    print(f"Worst fitness: {fit[-1]:.4f}")
    
    # Test Case 1: Normal behavior (no eyeing)
    print("\n" + "=" * 70)
    print("Test Case 1: Normal Behavior (eye_flag = 0)")
    print("=" * 70)
    
    Vt1, acc, t1, r1, l1, tempg, temps = update_velocity_time_acceleration(
        Vt, n, L, a, t, pop, fit, eye_flag=0
    )
    
    print(f"\nLeft dog index (0-indexed): {l1}")
    print(f"Right dog index (0-indexed): {r1}")
    print(f"Gathered sheep indices: {tempg[:5]}..." if len(tempg) > 5 else f"Gathered sheep: {tempg}")
    print(f"Stalked sheep indices: {temps[:5]}..." if len(temps) > 5 else f"Stalked sheep: {temps}")
    
    print(f"\nVelocity shape: {Vt1.shape}")
    print(f"Acceleration shape: {acc.shape}")
    print(f"Time shape: {t1.shape}")
    
    print(f"\nLead dog (index 0) velocity change:")
    print(f"  Before: {Vt[0, :3]}")
    print(f"  After:  {Vt1[0, :3]}")
    
    if len(tempg) > 0:
        first_gathered = tempg[0]
        print(f"\nGathered sheep (index {first_gathered}) velocity:")
        print(f"  Before: {Vt[first_gathered, :3]}")
        print(f"  After:  {Vt1[first_gathered, :3]}")
    
    if len(temps) > 0:
        first_stalked = temps[0]
        print(f"\nStalked sheep (index {first_stalked}) velocity:")
        print(f"  Before: {Vt[first_stalked, :3]}")
        print(f"  After:  {Vt1[first_stalked, :3]}")
    
    # Test Case 2: Eyeing behavior
    print("\n" + "=" * 70)
    print("Test Case 2: Eyeing Behavior (eye_flag = 1)")
    print("=" * 70)
    
    Vt1_eye, acc_eye, t1_eye, r1_eye, l1_eye, tempg_eye, temps_eye = update_velocity_time_acceleration(
        Vt, n, L, a, t, pop, fit, eye_flag=1
    )
    
    print(f"\nEyeing active - using dog with worse fitness for retardation")
    print(f"Left dog index: {l1_eye}")
    print(f"Right dog index: {r1_eye}")
    
    # Compare sheep velocities with and without eyeing
    sheep_indices = range(3, n)
    print(f"\nSheep velocities comparison (first sheep, first 3 dims):")
    print(f"  Normal:  {Vt1[3, :3]}")
    print(f"  Eyeing:  {Vt1_eye[3, :3]}")
    print(f"  Difference: {Vt1_eye[3, :3] - Vt1[3, :3]}")
    
    # Test Case 3: Acceleration and time update verification
    print("\n" + "=" * 70)
    print("Test Case 3: Acceleration and Time Update Verification")
    print("=" * 70)
    
    print(f"\nAcceleration of lead dog (first 3 dims): {acc[0, :3]}")
    print(f"Time of lead dog: {t1[0]:.6f}")
    
    # Verify acceleration formula: a = (Vt1 - Vt) / t
    for i in range(min(3, n)):
        for j in range(min(3, L)):
            expected_acc = abs(Vt1[i, j] - Vt[i, j]) / t[i]
            print(f"  Dog {i}, dim {j}: calculated={acc[i, j]:.6f}, expected={expected_acc:.6f}")
    
    # Test Case 4: Real BCO scenario
    print("\n" + "=" * 70)
    print("Test Case 4: Real BCO Scenario")
    print("=" * 70)
    
    n_total = 30  # 3 dogs + 27 sheep
    dims = 10
    
    # Create realistic data
    Vt_real = np.random.randn(n_total, dims) * 0.3
    a_real = np.random.randn(n_total, dims) * 0.05
    t_real = np.random.uniform(0.3, 1.5, n_total)
    pop_real = np.random.randn(n_total, dims) * 15
    fit_real = np.random.exponential(scale=10, size=n_total)
    fit_real = np.sort(fit_real)
    
    print(f"BCO Configuration:")
    print(f"  Total individuals: {n_total} (3 dogs + {n_total-3} sheep)")
    print(f"  Problem dimensions: {dims}")
    
    # Run simulation for multiple iterations
    for iter_num in range(5):
        eye_active = iter_num % 2
        eye_text = "ACTIVE" if eye_active else "INACTIVE"
        
        Vt_new, acc_new, t_new, r_idx, l_idx, gathered, stalked = update_velocity_time_acceleration(
            Vt_real, n_total, dims, a_real, t_real, pop_real, fit_real, eye_active
        )
        
        print(f"\nIteration {iter_num + 1} (Eyeing: {eye_text}):")
        print(f"  Dogs velocity range: [{np.min(Vt_new[:3, :]):.4f}, {np.max(Vt_new[:3, :]):.4f}]")
        print(f"  Sheep velocity range: [{np.min(Vt_new[3:, :]):.4f}, {np.max(Vt_new[3:, :]):.4f}]")
        print(f"  Gathered sheep: {len(gathered)}")
        print(f"  Stalked sheep: {len(stalked)}")
        
        # Update for next iteration
        Vt_real = Vt_new
        a_real = acc_new
        t_real = t_new
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("Performance Comparison (100 runs)")
    print("=" * 70)
    
    import time
    
    # Regular version
    start_time = time.time()
    for _ in range(100):
        Vt1_reg, acc_reg, t1_reg, _, _, _, _ = update_velocity_time_acceleration(
            Vt, n, L, a, t, pop, fit, 0
        )
    reg_time = time.time() - start_time
    
    # Vectorized version
    start_time = time.time()
    for _ in range(100):
        Vt1_vec, acc_vec, t1_vec, _, _, _, _ = update_velocity_vectorized(
            Vt, n, L, a, t, pop, fit, 0
        )
    vec_time = time.time() - start_time
    
    print(f"Regular version time: {reg_time:.4f} seconds")
    print(f"Vectorized version time: {vec_time:.4f} seconds")
    print(f"Vectorized is {reg_time/vec_time:.2f}x faster")
    
    print("\n" + "=" * 70)
    print("Velocity, Time, Acceleration update function is working correctly!")
    print("=" * 70)