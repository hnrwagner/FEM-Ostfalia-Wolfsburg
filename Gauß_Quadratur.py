import numpy as np
import matplotlib.pyplot as plt

# Beispielhafte Spannungs-Dehnungs-Daten (z. B. aus einem Zugversuch)
epsilon = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])  # Dehnung
sigma = np.array([0, 150, 280, 400, 450, 480])           # Spannung in MPa


# Ziel: Berechnung der Formänderungsenergie (Fläche unter der Spannungs-Dehnungs-Kurve) mit Gauß-Quadratur

# Gauß-Quadraturpunkte und Gewichte für 2-Punkt-Gauß-Quadratur auf [-1, 1]
gauss_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])  # Abszissen
weights = np.array([1, 1])  # Gewichte

# Berechnung der Formänderungsenergie
U_gauss = 0  # Initialisierung der Formänderungsenergie

# Iteration über jedes Intervall in der Spannungs-Dehnungs-Kurve
for i in range(len(epsilon) - 1):
    # Intervallgrenzen
    a = epsilon[i]
    b = epsilon[i + 1]
    h = b - a  # Intervallbreite

    # Transformation der Gauß-Punkte auf das Intervall [a, b]
    mapped_points = 0.5 * (b - a) * gauss_points + 0.5 * (a + b)

    # Berechnung der Spannungswerte an den transformierten Gauß-Punkten
    sigma_mapped = np.interp(mapped_points, epsilon, sigma)  # Interpolation der Spannungswerte

    # Gauß-Quadratur über das Intervall
    U_gauss += np.sum(0.5 * (b - a) * weights * sigma_mapped)

# Ausgabe der berechneten Formänderungsenergie
U_gauss


# Ausgabe der berechneten Formänderungsenergie
print(f"Formänderungsenergie U: {U_gauss} MPa")

# Plotten des Spannungs-Dehnungs-Diagramms
plt.figure(figsize=(8, 6))
plt.plot(epsilon, sigma, marker='o', linestyle='-', color='b', label='Spannungs-Dehnungs-Kurve')
plt.fill_between(epsilon, sigma, color="lightblue", alpha=0.3, label="Formänderungsenergie")

# Diagrammanpassungen
plt.title("Spannungs-Dehnungs-Diagramm")
plt.xlabel("Dehnung $\epsilon$")
plt.ylabel("Spannung $\sigma$ (MPa)")
plt.legend()
plt.grid(True)
plt.show()
