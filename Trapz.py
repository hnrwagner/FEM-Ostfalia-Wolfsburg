import numpy as np
import matplotlib.pyplot as plt

# Beispielhafte Spannungs-Dehnungs-Daten (z. B. aus einem Zugversuch)
epsilon = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])  # Dehnung
sigma = np.array([0, 150, 280, 400, 450, 480])           # Spannung in MPa




# Schrittweite h zwischen den Datenpunkten (angenommen konstant, für allgemeine Formel geeignet)
h = (epsilon[-1] - epsilon[0]) / (len(epsilon) - 1)

# Berechnung der Formänderungsenergie nach der allgemeinen Formel der Trapezregel
U = (h / 2) * (sigma[0] + 2 * np.sum(sigma[1:-1]) + sigma[-1])




# Ausgabe der berechneten Formänderungsenergie
print(f"Formänderungsenergie U: {U} MPa")

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
