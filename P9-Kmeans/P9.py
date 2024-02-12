import matplotlib.pyplot as plt
import numpy as np

# Función para calcular la distancia euclidiana
def distancia(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Generar datos aleatorios
np.random.seed(4020)
datos = np.random.rand(30, 2)

# Número de clusters
k = 2

# Inicialización de centroides aleatorios
centroides = datos[np.random.choice(len(datos), k, replace=False)]

# Número máximo de iteraciones
max_iter = 20

# Crear colores para cada cluster
colores = ['red', 'blue', 'green', 'purple', 'yellow']

# Crear subplots para mostrar las iteraciones
fig, axs = plt.subplots(5, 4, figsize=(12, 15))

for i in range(max_iter):
    axs[i // 4, i % 4].scatter(datos[:, 0], datos[:, 1], color='gray', s=30)
    
    asignaciones = []
    for punto in datos:
        distancias = [distancia(punto, c) for c in centroides]
        cluster = np.argmin(distancias)
        asignaciones.append(cluster)

    asignaciones = np.array(asignaciones)

    # Mostrar los puntos por cluster con diferentes colores
    for j in range(k):
        puntos_cluster = datos[asignaciones == j]
        axs[i // 4, i % 4].scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], color=colores[j], s=30)

    axs[i // 4, i % 4].scatter(centroides[:, 0], centroides[:, 1], color='black', marker='x', s=80)

    # Actualizar centroides
    for j in range(k):
        puntos_cluster = datos[asignaciones == j]
        if len(puntos_cluster) > 0:
            centroides[j] = np.mean(puntos_cluster, axis=0)

# Quitar números de las figuras
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Mostrar la gráfica final
plt.tight_layout()
plt.show()
