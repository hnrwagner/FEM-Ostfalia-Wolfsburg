import numpy as np
import matplotlib.pyplot as plt

# Beispielhafte Spannungs-Dehnungs-Daten (z. B. aus einem Zugversuch)
epsilon = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])  # Dehnung
sigma = np.array([0, 150, 280, 400, 450, 480])           # Spannung in MPa

# Berechnung der Ableitung der Spannungs-Dehnungs-Kurve mit zentralen Differenzen
d_sigma_d_epsilon = np.zeros(len(sigma))  # Initialisierung des Ableitungsvektors

# Zentrale Differenzen für alle inneren Punkte
for i in range(1, len(epsilon) - 1):
    d_sigma_d_epsilon[i] = (sigma[i + 1] - sigma[i - 1]) / (epsilon[i + 1] - epsilon[i - 1])

# Vorwärtsdifferenz für den ersten Punkt
d_sigma_d_epsilon[0] = (sigma[1] - sigma[0]) / (epsilon[1] - epsilon[0])

# Rückwärtsdifferenz für den letzten Punkt
d_sigma_d_epsilon[-1] = (sigma[-1] - sigma[-2]) / (epsilon[-1] - epsilon[-2])

# Erstellen der Diagramme
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Spannungs-Dehnungs-Diagramm
ax1.plot(epsilon, sigma, marker='o', linestyle='-', color='b', label='Spannungs-Dehnungs-Kurve')
ax1.fill_between(epsilon, sigma, color="lightblue", alpha=0.3, label="Formänderungsenergie")
ax1.set_title("Spannungs-Dehnungs-Diagramm")
ax1.set_xlabel("Dehnung $\epsilon$")
ax1.set_ylabel("Spannung $\sigma$ (MPa)")
ax1.legend()
ax1.grid(True)

# Plot 2: Ableitung der Spannungs-Dehnungs-Kurve
ax2.plot(epsilon, d_sigma_d_epsilon, marker='o', linestyle='-', color='r', label="dσ/dε")
ax2.set_title("Ableitung der Spannungs-Dehnungs-Kurve")
ax2.set_xlabel("Dehnung $\epsilon$")
ax2.set_ylabel("Steifigkeit dσ/dε (MPa)")
ax2.legend()
ax2.grid(True)

# Anzeigen der Diagramme
plt.tight_layout()
plt.show()
