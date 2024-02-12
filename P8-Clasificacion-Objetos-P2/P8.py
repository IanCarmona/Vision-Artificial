import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

carpeta_imagenes = './img/'

archivos_imagenes = os.listdir(carpeta_imagenes)

archivos_imagenes = [archivo for archivo in archivos_imagenes if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

try:
    indice_seleccionado = int(input(f'Ingrese el índice de la imagen (1-{len(archivos_imagenes)}): '))
    if 1 <= indice_seleccionado <= len(archivos_imagenes):
        # Cargar la imagen seleccionada
        imagen = cv2.imread(os.path.join(carpeta_imagenes, archivos_imagenes[indice_seleccionado - 1]))

        # Mostrar la imagen
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.title(f'Imagen {indice_seleccionado}')
        plt.show()
    else:
        print('Índice fuera de rango.')
except ValueError:
    print('Por favor, ingrese un número válido.')

# Convierte la imagen a escala de grises
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Aplica umbralización para obtener una imagen binaria
_, imagen_binaria = cv2.threshold(imagen_gris, 128, 255, cv2.THRESH_BINARY)

# Mostrar las tres imágenes en una ventana
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

plt.subplot(1, 3, 2)
plt.imshow(imagen_gris, cmap='gray')
plt.title('Imagen en Escala de Grises')

plt.subplot(1, 3, 3)
plt.imshow(imagen_binaria, cmap='gray')
plt.title('Imagen Binaria')

plt.show()

# Kernel para erosión y dilatación (5x5)
kernel1 = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=np.uint8)

kernel2 = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.uint8)

kernel3 = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
], dtype=np.uint8)

kernel4 = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 1]
], dtype=np.uint8)


def erosion(imagen, kernel):
    m, n = imagen.shape
    result = np.zeros((m, n), dtype=np.uint8)

    kh, kw = kernel.shape
    kh_half, kw_half = kh // 2, kw // 2

    for i in range(kh_half, m - kh_half):
        for j in range(kw_half, n - kw_half):
            # Comparar con el kernel invertido
            if np.all(imagen[i - kh_half:i + kh_half + 1, j - kw_half:j + kw_half + 1] == 255 * kernel[::-1, ::-1]):
                result[i, j] = 255
    img_er = cv2.erode(imagen, kernel, iterations=1)
    return result,img_er

# Función para realizar dilatación manualmente con kernel de tamaño n x n
def dilatacion(imagen, kernel):
    m, n = imagen.shape
    result = np.zeros((m, n), dtype=np.uint8)

    kh, kw = kernel.shape
    kh_half, kw_half = kh // 2, kw // 2

    for i in range(kh_half, m - kh_half):
        for j in range(kw_half, n - kw_half):
            if np.any(imagen[i - kh_half:i + kh_half + 1, j - kw_half:j + kw_half + 1] * kernel):
                result[i, j] = 255
    return result

# Función para invertir los píxeles de una imagen binaria manualmente
def invertir_imagen(imagen):
    m, n = imagen.shape
    imagen_invertida = np.zeros((m, n), dtype=np.uint8)

    for i in range(m):
        for j in range(n):
            # Invertir los valores de los píxeles
            imagen_invertida[i, j] = 255 - imagen[i, j]

    return imagen_invertida

# Aplicar erosión y dilatación a la imagen binaria
imagen_erosionada_1,img_er1 = erosion(imagen_binaria, kernel1)
imagen_erosionada_2,img_er2 = erosion(imagen_binaria, kernel2)
imagen_erosionada_3,img_er3 = erosion(imagen_binaria, kernel3)
imagen_erosionada_4,img_er4 = erosion(imagen_binaria, kernel4)

plt.figure(figsize = (10,5))
plt.subplot(3, 3, 2)
plt.imshow(cv2.cvtColor(imagen_binaria, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(img_er1, cv2.COLOR_BGR2RGB))
plt.title('Filtro "X"')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(cv2.cvtColor(img_er2, cv2.COLOR_BGR2RGB))
plt.title('Filtro "O"')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(cv2.cvtColor(img_er3, cv2.COLOR_BGR2RGB))
plt.title('Filtro "-"')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(cv2.cvtColor(img_er4, cv2.COLOR_BGR2RGB))
plt.title('Filtro "L"')
plt.axis('off')

plt.tight_layout()
plt.show()

# Listas para almacenar imágenes erosionadas y bordes
img_erosionadas = [img_er1, img_er2, img_er3, img_er4]
kernels = [kernel1, kernel2, kernel3, kernel4]

# Crear una figura de 4x4 para mostrar las imágenes de erosión y los bordes
plt.figure(figsize=(12, 12))

# Iterar sobre cada kernel y procesar las imágenes
for i in range(4):
    # Obtener la imagen erosionada y su kernel actual
    img_er = img_erosionadas[i]
    current_kernel = kernels[i]
    
    # Mostrar la imagen original
    plt.subplot(4, 4, i*4 + 1)
    plt.imshow(imagen_binaria, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Mostrar la imagen erosionada actual
    plt.subplot(4, 4, i*4 + 2)
    plt.imshow(cv2.cvtColor(img_er, cv2.COLOR_BGR2RGB))
    plt.title(f'Filtro {i+1}')
    plt.axis('off')
    
    # Invertir los píxeles de la imagen erosionada actual
    imagen_erosionada_invertida_manual = invertir_imagen(img_er)
    
    # Mostrar la imagen erosionada invertida actual
    plt.subplot(4, 4, i*4 + 3)
    plt.imshow(imagen_erosionada_invertida_manual, cmap='gray')
    plt.title(f'Invertida {i+1}')
    plt.axis('off')
    
    # Realizar la operación AND manualmente entre la imagen binaria original y la erosionada invertida actual
    bordes_manual = np.zeros_like(imagen_binaria)
    
    for x in range(imagen_binaria.shape[0]):
        for y in range(imagen_binaria.shape[1]):
            bordes_manual[x, y] = imagen_binaria[x, y] & imagen_erosionada_invertida_manual[x, y]
    
    # Mostrar el resultado de los bordes
    plt.subplot(4, 4, i*4 + 4)
    plt.imshow(bordes_manual, cmap='gray')
    plt.title(f'Bordes {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()
