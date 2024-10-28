import numpy as np
import matplotlib.pyplot as plt

def newmark_beta(M, C, K, F, u0, v0, a0, dt, steps, beta=0.25, gamma=0.5):
    """
    Newmark-beta method for solving a dynamic system.
    
    Parameters:
    M (ndarray): Mass matrix (6x6).
    C (ndarray): Damping matrix (6x6).
    K (ndarray): Stiffness matrix (6x6).
    F (ndarray): Force matrix over time (steps x 6).
    u0 (ndarray): Initial displacement vector (6,).
    v0 (ndarray): Initial velocity vector (6,).
    a0 (ndarray): Initial acceleration vector (6,).
    dt (float): Time step.
    steps (int): Number of time steps.
    beta (float): Newmark-beta parameter.
    gamma (float): Newmark-gamma parameter.
    
    Returns:
    ndarray: Displacement, velocity, and acceleration over time.
    """
    # Initialize displacement, velocity, and acceleration arrays
    u = np.zeros((steps, len(u0)))
    v = np.zeros((steps, len(u0)))
    a = np.zeros((steps, len(u0)))
    
    # Set initial conditions
    u[0, :] = u0
    v[0, :] = v0
    a[0, :] = a0
    
    # Effective stiffness matrix (constant for linear system)
    K_eff = M + gamma * dt * C + beta * dt**2 * K

    # Time-stepping loop
    for n in range(1, steps):
        # Effective force vector at time step n+1
        F_eff = F[n, :] - C @ (v[n-1, :] + (1 - gamma) * dt * a[n-1, :]) - K @ (u[n-1, :] + dt * v[n-1, :] + 0.5 * (1 - 2 * beta) * dt**2 * a[n-1, :])

        # Solve for acceleration at time step n+1
        a[n, :] = np.linalg.solve(K_eff, F_eff)
        
        # Update velocity and displacement for time step n+1
        v[n, :] = v[n-1, :] + dt * ((1 - gamma) * a[n-1, :] + gamma * a[n, :])
        u[n, :] = u[n-1, :] + dt * v[n-1, :] + 0.5 * dt**2 * ((1 - 2 * beta) * a[n-1, :] + 2 * beta * a[n, :])

    return u, v, a

# Define system matrices (6x6 example)
K = np.array([
    [10, -2, 0, 0, 0, 0],
    [-2, 12, -3, 0, 0, 0],
    [0, -3, 15, -4, 0, 0],
    [0, 0, -4, 18, -5, 0],
    [0, 0, 0, -5, 20, -6],
    [0, 0, 0, 0, -6, 22]
])

M = np.eye(6) * 2   # Mass matrix (diagonal)
C = np.eye(6) * 0.1  # Damping matrix (diagonal)

# Initial conditions (system at rest initially)
u0 = np.zeros(6)
v0 = np.zeros(6)
a0 = np.zeros(6)

# Time parameters
dt = 0.01  # Time step size
steps = 100  # Number of time steps

# Constant external force applied (for simplicity)
F = np.ones((steps, 6)) * 10  # Constant force on each degree of freedom over time

# Solve the system with Newmark-beta method
u, v, a = newmark_beta(M, C, K, F, u0, v0, a0, dt, steps, beta=0.25, gamma=0.5)

# Time vector for plotting
time = np.linspace(0, dt * (steps - 1), steps)

# Plot Displacement over time
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(time, u[:, i], label=f"DOF {i+1}")
plt.title("Displacement over Time for Each Degree of Freedom")
plt.xlabel("Time [s]")
plt.ylabel("Displacement")
plt.legend()
plt.grid(True)
plt.show()

# Plot Velocity over time
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(time, v[:, i], label=f"DOF {i+1}")
plt.title("Velocity over Time for Each Degree of Freedom")
plt.xlabel("Time [s]")
plt.ylabel("Velocity")
plt.legend()
plt.grid(True)
plt.show()

# Plot Acceleration over time
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(time, a[:, i], label=f"DOF {i+1}")
plt.title("Acceleration over Time for Each Degree of Freedom")
plt.xlabel("Time [s]")
plt.ylabel("Acceleration")
plt.legend()
plt.grid(True)
plt.show()
