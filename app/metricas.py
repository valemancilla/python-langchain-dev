import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # necesario para el gráfico 3D
# ------------------------------------------------
# 2. Definir los vectores de ejemplo
# ------------------------------------------------

# Dimensiones: [programación, finanzas, creatividad]

# A = curso básico de Python: mucha programación,
# casi nada de finanzas y creatividad
A = np.array([0.9, 0.1, 0.1])  # Curso A: "Introducción a Python"

# B = curso de Python aplicado a finanzas:
# mucha programación y muchas finanzas
B = np.array([0.9, 0.9, 0.1])  # Curso B: "Python avanzado para Finanzas"

# C = curso que no tiene relación con Python ni finanzas
C = np.array([0.1, 0.1, 1.0])  # Curso C: "Diseño Gráfico con Canva"
# ------------------------------------------------
# 3. Visualizar los vectores en un gráfico 3D (versión mejorada)
# ------------------------------------------------

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Dibujar los vectores desde el origen
ax.quiver(
    0, 0, 0, A[0], A[1], A[2],
    color='blue', linewidth=2, arrow_length_ratio=0.1,
    label='Introducción a Python'
)

ax.quiver(
    0, 0, 0, B[0], B[1], B[2],
    color='green', linewidth=2, arrow_length_ratio=0.1,
    label='Python avanzado para Finanzas'
)

ax.quiver(
    0, 0, 0, C[0], C[1], C[2],
    color='orange', linewidth=2, arrow_length_ratio=0.1,
    label='Diseño Gráfico con Canva'
)

# Limitar ejes (un poco más amplios para que no se corten)
ax.set_xlim([0, 1.1])
ax.set_ylim([0, 1.1])
ax.set_zlim([0, 1.1])

# Etiquetas de ejes (con tipografía más grande para mejor lectura)
ax.set_xlabel('Programación / Python', fontsize=11, labelpad=10)
ax.set_ylabel('Finanzas / Negocio', fontsize=11, labelpad=10)
ax.set_zlabel('Creatividad / Diseño', fontsize=11, labelpad=15)

# Aumentar el espacio del eje Z (importante para que no quede “aplastado”)
ax.set_box_aspect([1, 1, 1.2])  # Relación de aspecto X:Y:Z
ax.view_init(elev=25, azim=40)  # Elevar un poco la cámara y girar

# Agregar título y leyenda
ax.set_title(
    'Representación 3D de tres cursos como vectores',
    fontsize=13,
    pad=15
)
ax.legend(loc='upper left')

plt.tight_layout()
output_path = Path(__file__).with_name("metricas_vectores_3d.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------
# 4. Calcular las tres métricas de similitud
# ------------------------------------------------

# 4.1. Similitud del Coseno (Cosine Similarity)
# Mide el ángulo entre dos vectores (proximidad de dirección)

cos_AB = np.dot(A, B) / (norm(A) * norm(B))
cos_AC = np.dot(A, C) / (norm(A) * norm(C))
cos_BC = np.dot(B, C) / (norm(B) * norm(C))

# ------------------------------------------------
# Mostrar los resultados
# ------------------------------------------------

print("======= RESULTADOS =======")
print("\n Similitud del Coseno")
print(f"A vs B: {cos_AB:.3f}")
print(f"A vs C: {cos_AC:.3f}")
print(f"B vs C: {cos_BC:.3f}")

# 4.2. Distancia Euclidiana (L2 Norm)
# Mide la distancia geométrica absoluta entre los puntos

l2_AB = norm(A - B)
l2_AC = norm(A - C)
l2_BC = norm(B - C)

# ------------------------------------------------
# Mostrar los resultados
# ------------------------------------------------

print("======= RESULTADOS =======")
print("\n Distancia Euclidiana (L2 Norm)")
print(f"A vs B: {l2_AB:.3f}")
print(f"A vs C: {l2_AC:.3f}")
print(f"B vs C: {l2_BC:.3f}")

# 4.3. Producto Interno (Dot Product)
# Mide la proyección de un vector sobre otro

dot_AB = np.dot(A, B)
dot_AC = np.dot(A, C)
dot_BC = np.dot(B, C)

# ------------------------------------------------
# Mostrar los resultados
# ------------------------------------------------

print("======= RESULTADOS =======")
print("\n Producto Interno (Dot Product)")
print(f"A vs B: {dot_AB:.3f}")
print(f"A vs C: {dot_AC:.3f}")
print(f"B vs C: {dot_BC:.3f}")