# Solicitar al usuario que elija un índice de imagen
selected_index = int(input("Ingrese el índice de la imagen que desea ver (rango de 1 a 100): ")) - 1

# Obtener el número de objetos en la imagen seleccionada
selected_image_path = output_images[selected_index]
selected_image_name = os.path.basename(selected_image_path)
selected_image_index = images.index(os.path.join(image_folder, selected_image_name))
selected_image_objects = contador_a[selected_image_index]

# Leer la imagen seleccionada
selected_image = cv2.imread(selected_image_path, cv2.IMREAD_GRAYSCALE)

# Aplicar umbral o cualquier otro procesamiento que necesites para detectar contornos
_, binary_image = cv2.threshold(selected_image, 128, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la imagen binaria
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Contadores por tipo de objeto
count_tornillo = 0
count_armella = 0
count_pato = 0
count_rondana = 0
count_llave = 0

# Iterar sobre los contornos
for contour in contours:
    # Calcular el área del objeto
    area = cv2.contourArea(contour)

    # Filtro para eliminar objetos pequeños (ajusta el valor según tus necesidades)
    if area < 100:
        continue

    # Clasificar el objeto según su tipo
    if new_classifications[selected_index] == 1:
        count_llave += 1
    elif new_classifications[selected_index] == 2:
        count_tornillo += 1
    elif new_classifications[selected_index] == 3:
        count_rondana += 1
    elif new_classifications[selected_index] == 4:
        count_armella += 1
    elif new_classifications[selected_index] == 5:
        count_pato += 1

# Mostrar el número de objetos por tipo en la imagen seleccionada
print(f"Número de tornillos en la imagen {selected_image_name}: {count_tornillo}")
print(f"Número de armellas en la imagen {selected_image_name}: {count_armella}")
print(f"Número de patos en la imagen {selected_image_name}: {count_pato}")
print(f"Número de rondanas en la imagen {selected_image_name}: {count_rondana}")
print(f"Número de llaves hexagonales en la imagen {selected_image_name}: {count_llave}")


# Verificar que el índice esté dentro del rango válido
if 0 <= selected_index < len(output_images):
    # Cargar y mostrar la imagen seleccionada
    selected_image = cv2.imread(output_images[selected_index])
    cv2.imshow(f"Imagen seleccionada - Índice: {selected_index + 1}", selected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Índice fuera de rango. Por favor, ingrese un número válido.")



