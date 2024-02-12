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
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
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

# Aplicar erosión y dilatación a la imagen binaria
imagen_erosionada_1,img_er1 = erosion(imagen_binaria, kernel1)
imagen_dilatada_1 = dilatacion(imagen_binaria, kernel1)

imagen_erosionada_2,img_er2 = erosion(imagen_binaria, kernel2)
imagen_dilatada_2 = dilatacion(imagen_binaria, kernel2)

imagen_erosionada_3,img_er3 = erosion(imagen_binaria, kernel3)
imagen_dilatada_3 = dilatacion(imagen_binaria, kernel3)

imagen_erosionada_4,img_er4 = erosion(imagen_binaria, kernel4)
imagen_dilatada_4 = dilatacion(imagen_binaria, kernel4)



print("1) Erosionar: ")
print("2) Dilatar: ")
eleccion = int(input("Ingrese la opcion: "))

if eleccion == 1:
    plt.figure()
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
    plt.title('Filtro "."')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

elif eleccion == 2:
    plt.figure()
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(imagen_binaria, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(imagen_dilatada_1, cv2.COLOR_BGR2RGB))
    plt.title('Filtro "X"')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(imagen_dilatada_2, cv2.COLOR_BGR2RGB))
    plt.title('Filtro "O"')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(imagen_dilatada_3, cv2.COLOR_BGR2RGB))
    plt.title('Filtro "-"')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(cv2.cvtColor(imagen_dilatada_4, cv2.COLOR_BGR2RGB))
    plt.title('Filtro "."')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("Opcion invalida")



