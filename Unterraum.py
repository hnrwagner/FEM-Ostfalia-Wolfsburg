import numpy as np
from scipy.linalg import qr

# Definition der 6x6 Steifigkeitsmatrix K und der Massenmatrix M
K = np.array([
    [10, -2, 0, 0, 0, 0],
    [-2, 12, -3, 0, 0, 0],
    [0, -3, 15, -4, 0, 0],
    [0, 0, -4, 18, -5, 0],
    [0, 0, 0, -5, 20, -6],
    [0, 0, 0, 0, -6, 22]
])

M = np.array([
    [2, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0],
    [0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 2]
])

# Parameter
tol = 1e-6          # Toleranz für Konvergenz
max_iterations = 100 # Maximale Anzahl der Iterationen
p = 2                # Anzahl der zu berechnenden Eigenwerte (Unterraumdimension)

# Schritt 1: Initialisierung des Startunterraums mit zufälligen Vektoren
np.random.seed(0)
X = np.random.rand(K.shape[0], p)

for iteration in range(max_iterations):
    # Schritt 2: Berechne Y = K^{-1} M X
    # Statt die Inverse zu berechnen, lösen wir das Gleichungssystem K * Y = M * X
    Y = np.linalg.solve(K, M @ X)

    # Schritt 3: Orthogonalisierung von Y (QR-Zerlegung)
    Q, R = qr(Y, mode='economic')

    # Schritt 4: Reduziertes Eigenwertproblem
    T = Q.T @ K @ Q
    S = Q.T @ M @ Q

    # Eigenwerte und Eigenvektoren des reduzierten Problems
    eigenvalues, Z = np.linalg.eig(np.linalg.solve(S, T))

    # Ritz-Eigenvektoren des Originalproblems
    X_new = Q @ Z

    # Abbruchkriterium: Prüfe die Konvergenz der Eigenvektoren
    if np.linalg.norm(X - X_new) < tol:
        print("Konvergenz erreicht nach", iteration + 1, "Iterationen")
        break

    # Aktualisiere den Unterraum für die nächste Iteration
    X = X_new

# Sortiere die Eigenwerte und die zugehörigen Eigenvektoren
sorted_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = X[:, sorted_indices]

# Ausgabe der berechneten Eigenwerte und Eigenvektoren
print("Berechnete Eigenwerte:", eigenvalues[:p])
print("Berechnete Eigenvektoren (spaltenweise):\n", eigenvectors[:, :p])
