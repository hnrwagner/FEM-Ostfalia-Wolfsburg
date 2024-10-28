import numpy as np
from scipy.linalg import lu

# 6x6 Matrix K
K = np.array([
    [10, -2, 0, 0, 0, 0],
    [-2, 12, -3, 0, 0, 0],
    [0, -3, 15, -4, 0, 0],
    [0, 0, -4, 18, -5, 0],
    [0, 0, 0, -5, 20, -6],
    [0, 0, 0, 0, -6, 22]
])

# Kraftvektor F
F = np.array([1, 2, 3, 4, 5, 6])

# LU-Zerlegung der Matrix K
P, L, U = lu(K)

# Funktion für das Vorwärtseinsetzen (L * y = F)
def forward_substitution(L, b):
    """Löst L * y = b durch Vorwärtseinsetzen."""
    y = np.zeros_like(b, dtype=np.double)
    for i in range(len(b)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

# Funktion für das Rückwärtseinsetzen (U * u = y)
def backward_substitution(U, y):
    """Löst U * u = y durch Rückwärtseinsetzen."""
    u = np.zeros_like(y, dtype=np.double)
    for i in range(len(y) - 1, -1, -1):
        u[i] = (y[i] - np.dot(U[i, i + 1:], u[i + 1:])) / U[i, i]
    return u

# Schritt 1: Vorwärtseinsetzen, um y zu berechnen
y = forward_substitution(L, F)

# Schritt 2: Rückwärtseinsetzen, um u zu berechnen
u = backward_substitution(U, y)

print("Lösung des Unbekanntenvektors u:", u)
