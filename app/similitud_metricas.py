# ==============================================
# SESIÓN: ILUSTRACIÓN DE MÉTRICAS DE SIMILITUD
# Ejemplo con tres productos de ahorro / inversión
# ==============================================

# ----------------------------------------------
# 1. Importar bibliotecas necesarias
# ----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------
# 2. Definir los vectores de ejemplo
# ----------------------------------------------
# Dimensiones:
# [Liquidez y simplicidad, Seguridad del capital,
#  Enfoque agresivo de crecimiento]

# A: Cuenta de ahorro digital
A = np.array([0.85, 0.75, 0.20])

# B: Depósito a plazo fijo
B = np.array([0.55, 0.70, 0.25])

# C: Fondo de inversión agresivo
C = np.array([0.25, 0.40, 1.00])

# ----------------------------------------------
# 3. Visualización 3D
# ----------------------------------------------
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Vector A
ax.quiver(
    0, 0, 0,
    A[0], A[1], A[2],
    color='blue',
    linewidth=2,
    arrow_length_ratio=0.1,
    label='Cuenta de ahorro digital'
)

# Vector B
ax.quiver(
    0, 0, 0,
    B[0], B[1], B[2],
    color='green',
    linewidth=2,
    arrow_length_ratio=0.1,
    label='Depósito a plazo fijo'
)

# Vector C
ax.quiver(
    0, 0, 0,
    C[0], C[1], C[2],
    color='orange',
    linewidth=2,
    arrow_length_ratio=0.1,
    label='Fondo de inversión agresivo'
)

# Límites
ax.set_xlim([0, 1.1])
ax.set_ylim([0, 1.1])
ax.set_zlim([0, 1.1])

# Etiquetas
ax.set_xlabel('Liquidez / Simplicidad')
ax.set_ylabel('Seguridad del capital')
ax.set_zlabel('Enfoque agresivo / Crecimiento')

# Vista similar a la imagen
ax.set_box_aspect([1, 1, 1.2])
ax.view_init(elev=25, azim=40)

# Título y leyenda
ax.set_title('Productos de inversión como vectores en 3D')
ax.legend(loc='upper left')

plt.tight_layout()

# Guardar imagen
output_path = Path(__file__).with_name(
    "productos_bancarios_3d.png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches='tight'
)

plt.close(fig)

# ----------------------------------------------
# 4. Similitud del Coseno
# ----------------------------------------------
cos_AB = np.dot(A, B) / (norm(A) * norm(B))
cos_AC = np.dot(A, C) / (norm(A) * norm(C))
cos_BC = np.dot(B, C) / (norm(B) * norm(C))

print("========= RESULTADOS =========")

print("\n🔹 Similitud del Coseno")
print(f"A vs B: {cos_AB:.3f}")
print(f"A vs C: {cos_AC:.3f}")
print(f"B vs C: {cos_BC:.3f}")

# ----------------------------------------------
# 5. Distancia Euclidiana
# ----------------------------------------------
l2_AB = norm(A - B)
l2_AC = norm(A - C)
l2_BC = norm(B - C)

print("\n🔹 Distancia Euclidiana (L2 Norm)")
print(f"A vs B: {l2_AB:.3f}")
print(f"A vs C: {l2_AC:.3f}")
print(f"B vs C: {l2_BC:.3f}")

# ----------------------------------------------
# 6. Producto Interno
# ----------------------------------------------
dot_AB = np.dot(A, B)
dot_AC = np.dot(A, C)
dot_BC = np.dot(B, C)

print("\n🔹 Producto Interno (Dot Product)")
print(f"A vs B: {dot_AB:.3f}")
print(f"A vs C: {dot_AC:.3f}")
print(f"B vs C: {dot_BC:.3f}")

print("\nImagen guardada como: productos_bancarios_3d.png")