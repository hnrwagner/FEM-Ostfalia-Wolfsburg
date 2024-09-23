import numpy as np
import matplotlib.pyplot as plt

# by Dr. Ronald Wagner for Ostfalia Wolfsburg - "FEM mit Labor"  23.09.2024

# Parameters


P = -2670000.0                           # N/mm  # Applied point load at free end (N)
E = 210000                              # N/mm**2  # Young's Modulus in MPa
L = 36000                               # mm  # Length of the tower (mm)
D = 5500
d = 5466
I = np.pi/64.0*(D**4-d**4)              # mm**4  # Moment of inertia (mm^4)
EI = E*I


# Element length
L_e = L / 2

# Local stiffness matrix for a beam element
def beam_element_stiffness(E, I, L_e):
    return (E * I / L_e**3) * np.array([
        [12, 6*L_e, -12, 6*L_e],
        [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12, -6*L_e, 12, -6*L_e],
        [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ])

# Assemble the global stiffness matrix
K_local = beam_element_stiffness(E, I, L_e)

# Global stiffness matrix for two elements
K_global = np.zeros((6, 6))

# Assemble the stiffness matrix
K_global[:4, :4] += K_local  # Element 1 contribution
K_global[2:, 2:] += K_local  # Element 2 contribution

# Apply boundary conditions (clamped at node 1, remove rows and columns for w1 and θ1)
K_reduced = K_global[2:, 2:]

# Load vector (force at the free end)
F = np.zeros(4)
F[-2] = P  # Load applied at node 3

# Solve for the displacements and rotations
d = np.linalg.solve(K_reduced, F)

# Insert boundary conditions back into the displacement vector
displacements = np.zeros(6)
displacements[2:] = d

# Extract nodal displacements and rotations
w2, theta2, w3, theta3 = displacements[2:]

# Shape functions
def shape_functions(x, L_e):
    N1 = 1 - 3*(x/L_e)**2 + 2*(x/L_e)**3
    N2 = x * (1 - 2*(x/L_e) + (x/L_e)**2)
    N3 = 3*(x/L_e)**2 - 2*(x/L_e)**3
    N4 = x * ((x/L_e) - (x/L_e)**2)
    return N1, N2, N3, N4

# Approximate deflection w(x) in element 1 and 2
def w_approx(x, element, w_i, theta_i, w_j, theta_j, L_e):
    N1, N2, N3, N4 = shape_functions(x, L_e)
    return N1 * w_i + N2 * theta_i + N3 * w_j + N4 * theta_j

# Define x values for plotting in each element
x1 = np.linspace(0, L_e, 50)
x2 = np.linspace(0, L_e, 50)

# Compute deflection for element 1 (0 <= x <= L/2)
w_elem1 = w_approx(x1, 1, 0, 0, w2, theta2, L_e)

# Compute deflection for element 2 (L/2 <= x <= L)
w_elem2 = w_approx(x2, 2, w2, theta2, w3, theta3, L_e)


w = []
x_data = []
w_2 = []
w_3 = []
w_4 = []

for x in range(0,L,1):

    w.append(P*L/(4*EI)*x**2)   # Ritz

    w_2.append(P/(2*EI)*x**2*(L-x/3)) # Direct

    w_3.append(P/(2*EI)*x**2*(L-x/3)) # Galerkin

    x_data.append(x) # x


 # FEM Beam Model

w_FEM = (0,w2,w3)
x_FEM = (0,L/2,L)
fig, ax = plt.subplots()

 # FEM ABAQUS Tower

w_FEM_RR = (0,-262.145)
x_FEM_RR = (0,L)

# Plotting the approximate solution for w(x)
plt.plot(x_data, w, '-',color='orange', label='Variational Method - Ritz Method')
ax.plot(x_data, w_2, 'r-', label='Direct Method - exact solution')
plt.plot(x_data, w_3, '--',color='green', label='Galerkin Method of Weighted Residuals')
plt.scatter(x_FEM_RR,w_FEM_RR, label="FEM Complex Model")
plt.scatter(x_FEM,w_FEM, label="FEM Beam Model")


plt.ylabel('Deflection [mm]')
plt.xlabel('Tower Height [mm]')

legend = ax.legend(loc='lower left', shadow=True)


plt.grid(True)
