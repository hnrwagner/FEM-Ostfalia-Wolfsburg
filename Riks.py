import numpy as np
from scipy.linalg import qr

# Definition des externen Kraftvektors (normiert für das Riks-Verfahren)
F_ext = np.array([1, 2, 3, 4, 5, 6])

# Parameter für das Riks-Verfahren
tol = 1e-6         # Toleranz für Konvergenz
max_iterations = 50  # Maximale Anzahl der Iterationen
s = 0.1            # Bogenlängeninkrement

# Startwerte
u = np.zeros(6)    # Anfangsverschiebung
lambda_load = 0    # Anfangslastfaktor

# Funktion zur Berechnung der Tangentsteifigkeitsmatrix K(u)
def tangent_stiffness_matrix(u):
    return np.array([
        [10 + 0.5 * u[0], -2, 0, 0, 0, 0],
        [-2, 12 + 0.3 * u[1], -3, 0, 0, 0],
        [0, -3, 15 + 0.4 * u[2], -4, 0, 0],
        [0, 0, -4, 18 + 0.2 * u[3], -5, 0],
        [0, 0, 0, -5, 20 + 0.1 * u[4], -6],
        [0, 0, 0, 0, -6, 22 + 0.3 * u[5]]
    ])

# Riks-Iteration
for iteration in range(max_iterations):
    # Berechne Tangentsteifigkeitsmatrix K(u) und internen Kraftvektor F_int = K(u) * u
    K_u = tangent_stiffness_matrix(u)
    F_int = K_u @ u

    # Restkraftvektor und Bogenlängenbedingung
    R = F_ext * lambda_load - F_int
    residual_norm = np.linalg.norm(R)
    
    # Überprüfe Konvergenz
    if residual_norm < tol:
        print("Konvergenz erreicht nach", iteration + 1, "Iterationen.")
        break

    # Erweitere Tangentmatrix und Restkraftvektor für Bogenlängenbedingung
    K_augmented = np.block([
        [K_u, -F_ext.reshape(-1, 1)],
        [2 * u.reshape(1, -1), np.zeros((1, 1))]
    ])
    
    R_augmented = np.concatenate((R, [s**2 - np.dot(u, u)]))

    # Löse das erweiterte Gleichungssystem
    delta_solution = np.linalg.solve(K_augmented, R_augmented)
    delta_u = delta_solution[:-1]
    delta_lambda = delta_solution[-1]

    # Aktualisiere Verschiebungen und Lastfaktor
    u += delta_u
    lambda_load += delta_lambda

# Ausgabe der berechneten Verschiebungen und des Lastfaktors
print("Berechnete Verschiebungen u:", u)
print("Berechneter Lastfaktor λ:", lambda_load)
