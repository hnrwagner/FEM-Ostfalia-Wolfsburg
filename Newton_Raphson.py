import numpy as np

# Definition des externen Kraftvektors
F_ext = np.array([1, 2, 3, 4, 5, 6])

# Parameter für das Newton-Raphson-Verfahren
tol = 1e-6         # Toleranz für Konvergenz
max_iterations = 50  # Maximale Anzahl der Iterationen

# Startwert für die Verschiebungen
u = np.zeros(6)

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

# Newton-Raphson-Iteration
for iteration in range(max_iterations):
    # Berechne Tangentsteifigkeitsmatrix K(u) und internen Kraftvektor F_int = K(u) * u
    K_u = tangent_stiffness_matrix(u)
    F_int = K_u @ u

    # Berechne Restkraftvektor R = F_ext - F_int
    R = F_ext - F_int

    # Überprüfe Konvergenz
    if np.linalg.norm(R) < tol:
        print("Konvergenz erreicht nach", iteration + 1, "Iterationen.")
        break

    # Löse für Inkrement Delta_u
    delta_u = np.linalg.solve(K_u, R)

    # Aktualisiere Verschiebungsvektor u
    u = u + delta_u

# Ausgabe der berechneten Verschiebungen
print("Berechnete Verschiebungen u:", u)
