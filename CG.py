import numpy as np

# Definition der 6x6 Matrix K (symmetrisch und positiv definit)
K = np.array([
    [10, -2, 0, 0, 0, 0],
    [-2, 12, -3, 0, 0, 0],
    [0, -3, 15, -4, 0, 0],
    [0, 0, -4, 18, -5, 0],
    [0, 0, 0, -5, 20, -6],
    [0, 0, 0, 0, -6, 22]
])

# Der Kraftvektor F
F = np.array([1, 2, 3, 4, 5, 6])

# Conjugate Gradient Methode
def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iterations=1000):
    """Löst Ax = b mit der Conjugate Gradient Methode."""
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)  # Startwert x0 oder Nullvektor
    r = b - A.dot(x)  # Initialer Fehlervektor
    p = r.copy()      # Initiale Suchrichtung
    rs_old = np.dot(r, r)

    for k in range(max_iterations):
        Ap = A.dot(p)
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        
        # Überprüfe Abbruchkriterium
        if np.sqrt(rs_new) < tol:
            return x, k + 1  # gibt Lösung und Anzahl der Iterationen zurück
        
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, max_iterations  # gibt Lösung nach max. Iterationen zurück

# Ausführen der CG-Methode mit dem Startwert x0 = [0, 0, 0, 0, 0, 0]
solution, iterations = conjugate_gradient(K, F)

print("Lösung des Unbekanntenvektors x:", solution)
print("Anzahl der Iterationen:", iterations)
