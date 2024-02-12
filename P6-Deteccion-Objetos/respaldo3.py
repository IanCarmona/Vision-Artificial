import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy  # Agrega la importación de la biblioteca copy
from PIL import Image

image_folder = "./img"
images = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]
center_x = []
center_y = []
area_a = []
perimetro_a = []
contador_a = []
w_a = []
h_a = []
tornillo = 0
rondana = 0
pato = 0
llave = 0
armella = 0

for image_path in images:
    contador = 0
    image = cv2.imread(
        image_path, cv2.IMREAD_GRAYSCALE
    )  # Cargar la imagen en escala de grises
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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
        w_a.append(w)
        h_a.append(h)
        area_a.append(area)
        perimetro_a.append(perimeter)

    # Agregar el contador de objetos de esta imagen a la lista
    contador_a.append(contador)

features = np.column_stack((area_a, perimetro_a, h_a, w_a))

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
# print(new_classifications)

# Gráfico de dispersión para visualizar los clusters
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)
ax.set_xlabel("Área")
ax.set_ylabel("Perímetro")
ax.set_zlabel("w_a")

plt.title("Gráfico de Dispersión de Clusters en 3D")
plt.show()


# Ahora vamos a colorear los contornos en las imágenes originales y guardarlas
output_folder = "./output"  # Carpeta de salida para las imágenes coloreadas

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

copia_lista = copy.deepcopy(
    new_classifications
)  # Utilizar las nuevas clasificaciones ordenadas


# Obtener la lista de imágenes coloreadas
output_images = [
    os.path.join(output_folder, image) for image in os.listdir(output_folder)
]


count_tornillo = 0
count_armella = 0
count_pato = 0
count_rondana = 0
count_llave = 0


print("1) Imagen Internet:")
print("2) Imagen de la base:")

eleccion_t = int(input("Eleccion del tipo de busqueda: "))

if eleccion_t == 2:
    
    # Solicitar al usuario que elija un índice de imagen
    selected_index = (
        int(input("Ingrese el índice de la imagen que desea ver (rango de 1 a 100): ")) - 1
        
    )
    
    
    # Obtener el número de objetos en la imagen seleccionada
    selected_image_path = output_images[selected_index]
    selected_image_name = os.path.basename(selected_image_path)
    selected_image_index = images.index(os.path.join(image_folder, selected_image_name))
    selected_image_objects = contador_a[selected_image_index]

    imagenes_buscador_tornillo = []
    imagenes_buscador_armella = []
    imagenes_buscador_patos = []
    imagenes_buscador_rondana = []
    imagenes_buscador_llave = []





    for i, image_path in enumerate(images):
        image = cv2.imread(image_path)  # Cargar la imagen a color

        # Crear una copia de la imagen original para colorear los contornos
        image_colored = image.copy()

        # Convertir la imagen a escala de grises para trabajar con contornos
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral o cualquier otro procesamiento que necesites para detectar contornos
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Encontrar contornos en la imagen binaria
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
            # text = f"A: {area:.1f}, P: {perimeter:.1f}"  # Formatea el texto

            # Colorear el contorno según la clasificación

            if i == selected_index:
                
                seleccionada = []
                seleccionada.append(image_path)
                
                if copia_lista[0] == 1:
                    text = "llave alen."
                    count_llave = count_llave + 1
                    imagenes_buscador_llave.append(image_path)

                elif copia_lista[0] == 2:
                    text = "tornillo"
                    count_tornillo = count_tornillo + 1
                    imagenes_buscador_tornillo.append(image_path)
                    
                elif copia_lista[0] == 3:
                    text = "rondana"
                    count_rondana = count_rondana + 1
                    imagenes_buscador_rondana.append(image_path)
                elif copia_lista[0] == 4:
                    text = "armella"
                    count_armella = count_armella + 1
                    imagenes_buscador_armella.append(image_path)
                elif copia_lista[0] == 5:
                    text = "pato"
                    count_pato = count_pato + 1
                    imagenes_buscador_patos.append(image_path)
            else:
                if copia_lista[0] == 1:
                    text = "llave alen."
                    imagenes_buscador_llave.append(image_path)
                elif copia_lista[0] == 2:
                    text = "tornillo"
                    imagenes_buscador_tornillo.append(image_path)
                elif copia_lista[0] == 3:
                    text = "rondana"
                    imagenes_buscador_rondana.append(image_path)
                elif copia_lista[0] == 4:
                    text = "armella"
                    imagenes_buscador_armella.append(image_path)
                elif copia_lista[0] == 5:
                    text = "pato"
                    

            cv2.putText(
                image_colored, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

            copia_lista.pop(0)

            # Dibujar el contorno coloreado en la imagen original
            cv2.drawContours(image_colored, [contour], -1, color, 2)

        # Guardar la imagen coloreada en la carpeta de salida
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image_colored)


    # Verificar que el índice esté dentro del rango válido
    if 0 <= selected_index < len(output_images):
        # Cargar y mostrar la imagen seleccionada
        selected_image = cv2.imread(output_images[selected_index])
        cv2.imshow(f"Imagen seleccionada - Índice: {selected_index + 1}", selected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Índice fuera de rango. Por favor, ingrese un número válido.")


    # Mostrar el número de objetos por tipo en la imagen seleccionada
    print(f"Número de tornillos en la imagen {selected_image_name}: {count_tornillo}")
    print(f"Número de armellas en la imagen {selected_image_name}: {count_armella}")
    print(f"Número de patos en la imagen {selected_image_name}: {count_pato}")
    print(f"Número de rondanas en la imagen {selected_image_name}: {count_rondana}")
    print(f"Número de llaves alen en la imagen {selected_image_name}: {count_llave}")


    print()
    print("1) Buscar Tornillos")
    print("2) Buscar Armellas")
    print("3) Buscar Patos")
    print("4) Buscar Rondanas")
    print("5) Buscar Llave Alen")
    eleccion = int(input("Ingrese el objeto que quiere buscar: "))


    def contar_figuras(image_paths):
        resultados = {}
        
        

        for image_path in image_paths:
            contador = 0

            # Leer la imagen
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Procesar la imagen para la detección de contornos
            _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Calcular el área del objeto
                area = cv2.contourArea(contour)

                # Filtro para eliminar objetos pequeños
                if area < 100:
                    continue

                # Incrementar el contador para este objeto
                contador += 1

            resultados[image_path] = contador

        return contador



    contador = contar_figuras(seleccionada)





    if eleccion == 1:
        if count_tornillo == 0:
            print("En la imagen que elejiste no hay tornillos")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_tornillo)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                    

            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 2:
        if count_armella == 0:
            print("En la imagen que elejiste no hay armellas")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_armella)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 3:
        if count_pato == 0:
            print("En la imagen que elejiste no hay patos")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_patos)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 4:
        if count_rondana == 0:
            print("En la imagen que elejiste no hay rondana")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_rondana)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 5:
        if count_llave == 0:
            print("En la imagen que elejiste no hay llave alen")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_llave)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    else:
        print("Opcion invalida")

elif eleccion_t == 3:
    print("No se encuentra en la base de datos la imagen")
    imagen = Image.open("./oso.png")

    # Mostrar la imagen usando plt.imshow()
    plt.imshow(imagen)
    plt.axis('off')  # Ocultar los ejes si no son necesarios
    plt.title('Título de la imagen')  # Agregar un título opcional
    plt.show()



else:
    selected_index = 3
    # Obtener el número de objetos en la imagen seleccionada
    selected_image_path = output_images[3]
    selected_image_name = os.path.basename(selected_image_path)
    selected_image_index = images.index(os.path.join(image_folder, selected_image_name))
    selected_image_objects = contador_a[selected_image_index]

    imagenes_buscador_tornillo = []
    imagenes_buscador_armella = []
    imagenes_buscador_patos = []
    imagenes_buscador_rondana = []
    imagenes_buscador_llave = []





    for i, image_path in enumerate(images):
        image = cv2.imread(image_path)  # Cargar la imagen a color

        # Crear una copia de la imagen original para colorear los contornos
        image_colored = image.copy()

        # Convertir la imagen a escala de grises para trabajar con contornos
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral o cualquier otro procesamiento que necesites para detectar contornos
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

        # Encontrar contornos en la imagen binaria
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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
            # text = f"A: {area:.1f}, P: {perimeter:.1f}"  # Formatea el texto

            # Colorear el contorno según la clasificación

            if i == selected_index:
                
                seleccionada = []
                seleccionada.append(image_path)
                
                if copia_lista[0] == 1:
                    text = "llave alen."
                    count_llave = count_llave + 1
                    imagenes_buscador_llave.append(image_path)

                elif copia_lista[0] == 2:
                    text = "tornillo"
                    count_tornillo = count_tornillo + 1
                    imagenes_buscador_tornillo.append(image_path)
                    
                elif copia_lista[0] == 3:
                    text = "rondana"
                    count_rondana = count_rondana + 1
                    imagenes_buscador_rondana.append(image_path)
                elif copia_lista[0] == 4:
                    text = "armella"
                    count_armella = count_armella + 1
                    imagenes_buscador_armella.append(image_path)
                elif copia_lista[0] == 5:
                    text = "pato"
                    count_pato = count_pato + 1
                    imagenes_buscador_patos.append(image_path)
            else:
                if copia_lista[0] == 1:
                    text = "llave alen."
                    imagenes_buscador_llave.append(image_path)
                elif copia_lista[0] == 2:
                    text = "tornillo"
                    imagenes_buscador_tornillo.append(image_path)
                elif copia_lista[0] == 3:
                    text = "rondana"
                    imagenes_buscador_rondana.append(image_path)
                elif copia_lista[0] == 4:
                    text = "armella"
                    imagenes_buscador_armella.append(image_path)
                elif copia_lista[0] == 5:
                    text = "pato"
                    

            cv2.putText(
                image_colored, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

            copia_lista.pop(0)

            # Dibujar el contorno coloreado en la imagen original
            cv2.drawContours(image_colored, [contour], -1, color, 2)

        # Guardar la imagen coloreada en la carpeta de salida
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image_colored)


    # Verificar que el índice esté dentro del rango válido
    if 0 <= selected_index < len(output_images):
        # Cargar y mostrar la imagen seleccionada
        selected_image = cv2.imread(output_images[selected_index])
        
    else:
        print("Índice fuera de rango. Por favor, ingrese un número válido.")


    # Mostrar el número de objetos por tipo en la imagen seleccionada
    print(f"Número de tornillos en la imagen {selected_image_name}: {count_tornillo}")
    print(f"Número de armellas en la imagen {selected_image_name}: {count_armella}")
    print(f"Número de patos en la imagen {selected_image_name}: {count_pato}")
    print(f"Número de rondanas en la imagen {selected_image_name}: {count_rondana}")
    print(f"Número de llaves alen en la imagen {selected_image_name}: {count_llave}")


    print()
    print("1) Buscar Tornillos")
    print("2) Buscar Armellas")
    print("3) Buscar Patos")
    print("4) Buscar Rondanas")
    print("5) Buscar Llave Alen")
    eleccion = int(input("Ingrese el objeto que quiere buscar: "))


    def contar_figuras(image_paths):
        resultados = {}
        
        

        for image_path in image_paths:
            contador = 0

            # Leer la imagen
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Procesar la imagen para la detección de contornos
            _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Calcular el área del objeto
                area = cv2.contourArea(contour)

                # Filtro para eliminar objetos pequeños
                if area < 100:
                    continue

                # Incrementar el contador para este objeto
                contador += 1

            resultados[image_path] = contador

        return contador



    contador = contar_figuras(seleccionada)





    if eleccion == 1:
        if count_tornillo == 0:
            print("En la imagen que elejiste no hay tornillos")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_tornillo)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                    

            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 2:
        if count_armella == 0:
            print("En la imagen que elejiste no hay armellas")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_armella)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 3:
        if count_pato == 0:
            print("En la imagen que elejiste no hay patos")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_patos)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 4:
        if count_rondana == 0:
            print("En la imagen que elejiste no hay rondana")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_rondana)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open(seleccionada[0])
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    elif eleccion == 5:
        if count_llave == 0:
            print("En la imagen que elejiste no hay llave alen")
        else:
            conjunto_sin_duplicados = set(imagenes_buscador_llave)
            mi_arreglo_sin_duplicados = list(conjunto_sin_duplicados)
            mi_arreglo_final = []
            
            for path in mi_arreglo_sin_duplicados:
                if(contar_figuras([path]) <= contador):
                    mi_arreglo_final.append(path)
                
                
            mi_arreglo_final.remove(seleccionada[0])
            
            # Crear una figura y los subplots
            
            # Crear subplots 2x2
            fig, axs = plt.subplots(2, 2)

            # Mostrar la imagen específica en el primer subplot
            imagen_especifica = Image.open("./allen.jpg")
            axs[0, 0].imshow(imagen_especifica)
            axs[0, 0].axis("off")  # Ocultar los ejes
            axs[0, 0].set_title("Imagen Específica")

            # Mostrar las otras tres imágenes en los subplots restantes
            for i, ruta in enumerate(mi_arreglo_final):
                imagen = Image.open(ruta)
                fila = (i + 1) // 2  # Comenzar desde el siguiente subplot (después del primero)
                columna = (i + 1) % 2
                axs[fila, columna].imshow(imagen)
                axs[fila, columna].axis("off")  # Ocultar los ejes
                axs[fila, columna].set_title(f"Imagen {i + 2}")  # El índice comienza desde 2

            # Ajustar el diseño de los subplots y mostrar la figura
            plt.tight_layout()
            plt.show()

    else:
        print("Opcion invalida")
    
    


    
    



