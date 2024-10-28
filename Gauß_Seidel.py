import numpy as np

# Definition der 6x6 Matrix K
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

# Gauß-Seidel-Iterative Methode
def gauss_seidel(A, b, x0=None, tol=1e-10, max_iterations=1000):
    """Löst Ax = b mit der Gauß-Seidel-Methode."""
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)  # Startwert x0 oder Nullvektor
    
    for k in range(max_iterations):
        x_old = x.copy()
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])      # Summe der vorherigen Werte
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])  # Summe der "alten" Werte
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Überprüfe Abbruchkriterium
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k + 1  # gibt Lösung und Anzahl der Iterationen zurück
    
    return x, max_iterations  # gibt Lösung nach max. Iterationen zurück

# Ausführen der Gauß-Seidel-Methode mit dem Startwert x0 = [0, 0, 0, 0, 0, 0]
solution, iterations = gauss_seidel(K, F)

print("Lösung des Unbekanntenvektors x:", solution)
print("Anzahl der Iterationen:", iterations)
