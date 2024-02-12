import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy  # Agrega la importación de la biblioteca copy

image_folder = './img'
images = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]
center_x = []
center_y = []
area_a = []
perimetro_a = []
contador_a = []

for image_path in images:
    contador = 0
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Cargar la imagen en escala de grises
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calcular el área del objeto
        area = cv2.contourArea(contour)

        # Filtro para eliminar objetos pequeños (ajusta el valor según tus necesidades)
        if area < 100:
            continue

        # Incrementar el contador para este objeto
        contador += 1

        # Calcular el perímetro del objeto
        perimeter = cv2.arcLength(contour, True)

        # Dibujar un rectángulo alrededor del objeto
        x, y, w, h = cv2.boundingRect(contour)

        center_x.append(x + w // 2)
        center_y.append(y + h // 2)
        area_a.append(area)
        perimetro_a.append(perimeter)

    # Agregar el contador de objetos de esta imagen a la lista
    contador_a.append(contador)
    
features = np.column_stack((area_a, perimetro_a))

# Convertir la lista de características a un arreglo numpy
features = np.array(features)

# Entrenar el modelo KMeans con 5 clusters
kmeans = KMeans(n_clusters=5)
kmeans.fit(features)

# Obtener las etiquetas de clase asignadas por KMeans
labels = kmeans.labels_

# Almacenar las etiquetas de clase en un nuevo arreglo para su clasificación
classifications = []

for label in labels:
    # Clasificar los objetos en base a los clusters
    classifications.append(label + 1)  # Agregar 1 para que las etiquetas vayan de 1 a 5

# Crear un diccionario que asocie clases originales con áreas
class_area_dict = dict(zip(classifications, area_a))

# Ordenar el diccionario por áreas de forma ascendente
sorted_class_area = sorted(class_area_dict.items(), key=lambda x: x[1])

# Crear un nuevo diccionario con las clases ordenadas
sorted_class_dict = {k: i + 1 for i, (k, _) in enumerate(sorted_class_area)}

# Asignar nuevas clases basadas en el orden ascendente de áreas
new_classifications = [sorted_class_dict[class_] for class_ in classifications]

# Imprimir las nuevas clasificaciones
print(new_classifications)

# Gráfico de dispersión para visualizar los clusters
plt.scatter(features[:, 0], features[:, 1], c=labels)
plt.title("Gráfico de Dispersión de Clusters")
plt.xlabel("Área")
plt.ylabel("Perímetro")
plt.show()

# Ahora vamos a colorear los contornos en las imágenes originales y guardarlas
output_folder = './output'  # Carpeta de salida para las imágenes coloreadas

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

copia_lista = copy.deepcopy(new_classifications)  # Utilizar las nuevas clasificaciones ordenadas

for image_path in images:
    image = cv2.imread(image_path)  # Cargar la imagen a color

    # Crear una copia de la imagen original para colorear los contornos
    image_colored = image.copy()

    # Convertir la imagen a escala de grises para trabajar con contornos
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral o cualquier otro procesamiento que necesites para detectar contornos
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Calcular el área del objeto
        area = cv2.contourArea(contour)

        # Filtro para eliminar objetos pequeños (ajusta el valor según tus necesidades)
        if area < 100:
            continue
        
        perimeter = cv2.arcLength(contour, True)
        
        # Colorear el contorno según la clasificación
        if copia_lista[0] == 1:
            color = (0, 0, 255)  # Rojo
        elif copia_lista[0] == 2:
            color = (0, 255, 0)  # Verde
        elif copia_lista[0] == 3:
            color = (255, 0, 0)  # Azul
        elif copia_lista[0] == 4:
            color = (0, 255, 255)  # Amarillo
        elif copia_lista[0] == 5:
            color = (255, 0, 255)  # Magenta
            
        # Dibujar el contorno coloreado en la imagen original
        cv2.drawContours(image_colored, [contour], -1, color, 2)
        x, y, w, h = cv2.boundingRect(contour)

        # Agregar texto con el área y el perímetro
        #text = f"A: {area:.1f}, P: {perimeter:.1f}"  # Formatea el texto
        
                # Colorear el contorno según la clasificación
        if copia_lista[0] == 1:
            text="palo"
        elif copia_lista[0] == 2:
            text="tornillo"
        elif copia_lista[0] == 3:
            text="tuerca"
        elif copia_lista[0] == 4:
            text="cosita redonda"
        elif copia_lista[0] == 5:
            text="otro circulo"
        
        
        cv2.putText(image_colored, text, (x , y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)

        copia_lista.pop(0)

        # Dibujar el contorno coloreado en la imagen original
        cv2.drawContours(image_colored, [contour], -1, color, 2)

    # Guardar la imagen coloreada en la carpeta de salida
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image_colored)
