clc;
clear;
close all

n_classes = input('Numero de clases: ');
r_classes = input('Numero de representantes por clase: ');

img = imread("peppers.png");
imshow(img);

class_representatives = cell(1, n_classes);

for index = 1:n_classes
    % Inicializar un arreglo para los representantes de esta clase
    class_representatives{index} = zeros(r_classes, 3);

    for rep_num = 1:r_classes
        % Espera a que el usuario haga un clic y registra las coordenadas
        [x, y] = ginput(1);  % Espera a que el usuario haga un clic

        % Obtiene el color del pixel en el punto seleccionado
        pixel_value = impixel(img, x, y);  % Obtiene el color del pixel
        class_representatives{index}(rep_num, :) = pixel_value;
        
        % Coloca un punto negro, pequeño y relleno en el punto seleccionado
        hold on;
        plot(x, y, 'w.', 'MarkerSize', 5);
        hold off;

        if rep_num == r_classes
            % Agregar etiqueta cerca del último representante
            label_offset = -10; % Puedes ajustar este valor para controlar la posición de la etiqueta
            text(x + label_offset, y + label_offset, ['Clase ' num2str(index)], 'Color', 'k');
        end
    end
    


    % Mostrar un mensaje cuando se completan los representantes de una clase
    disp(['Terminado de ingresar representantes para la Clase ' num2str(index)]);

    
end

for index = 1:n_classes
    disp(['Clase ' num2str(index) ':']);
    disp(class_representatives{index});
end

% Trasponer cada matriz de clase y guardarlas en un arreglo
transposed_class_matrices = cell(1, n_classes);

for index = 1:n_classes
    transposed_class_matrices{index} = class_representatives{index}';

    % Mostrar la matriz transpuesta
    disp(['Matriz transpuesta de Clase ' num2str(index) ':']);
    disp(transposed_class_matrices{index});
end

% Calcular la media para cada clase
media = cell(1, n_classes);
media_original = cell(1, n_classes);

for index = 1:n_classes
    % Calcular la media para esta clase utilizando la fórmula que proporcionaste
    %media{index} = (1/length(class_representatives{index})) * sum(class_representatives{index})';
    media{index} = (1/r_classes) * sum(class_representatives{index})';
    media_original{index} = (1/r_classes) * sum(class_representatives{index})';


    % Expandir la media para que tenga las mismas dimensiones que la matriz de clase
    media{index} = repmat(media{index}, 1, r_classes);
    
    % Mostrar la media de la Clase
    disp(['Media de la Clase ' num2str(index) ':']);
    disp(media{index});
    disp(media_original{index});
end

% Calcular la covarianza para cada clase manualmente
covariance = cell(1, n_classes);

for index = 1:n_classes
    % Restar la transpuesta de la clase con su respectiva media
    covariaza_matrix = transposed_class_matrices{index} - media{index};
    % Almacenar la covarianza en el arreglo
    covariance{index} = covariaza_matrix;
    
    % Mostrar la covarianza de la Clase
    disp(['Covarianza de la Clase ' num2str(index) ':']);
    disp(covariance{index});
end

% Calcular la transpuesta de las covarianzas y mostrarlas en un ciclo for aparte
covariance_transposed = cell(1, n_classes);

for index = 1:n_classes
    % Calcular la transpuesta de la covarianza
    covariance_transposed{index} = covariance{index}';
    
    % Mostrar la transpuesta de la covarianza de la Clase
    disp(['Transpuesta de la Covarianza de la Clase ' num2str(index) ':']);
    disp(covariance_transposed{index});
end

% Calcular "dato" para cada clase de forma iterativa
dato = cell(1, n_classes);

for index = 1:n_classes
    % Calcular "dato" utilizando la fórmula iterativa
    dato{index} = (1/r_classes) * (covariance{index} * covariance_transposed{index});
    
    % Mostrar "dato" de la Clase
    disp(['Dato de la Clase ' num2str(index) ':']);
    disp(dato{index});
end

% Calcular la inversa de cada matriz de dato (dato_inv) para cada clase
dato_inv = cell(1, n_classes);

for index = 1:n_classes
    % Calcular la inversa de la matriz de dato de la clase actual
    dato_inv{index} = inv(dato{index});
    
    % Mostrar la matriz inversa de la Clase
    disp(['Inversa de la matriz de dato de la Clase ' num2str(index) ':']);
    disp(dato_inv{index});
end

while true
% Solicitar al usuario hacer clic en la imagen para seleccionar el punto vector
    % Cambiar la figura activa a la imagen original
    disp('Por favor, haga clic en la imagen para seleccionar un punto como vector.');
    figure(1);  % Cambiar a la figura de la imagen original
    [x, y] = ginput(1);  % Esperar a que el usuario haga clic en la imagen
    
    % Volver a la figura de la gráfica 3D
    figure(2);  % Cambiar a la figura de la gráfica 3D

    % Obtener el color del pixel en el punto seleccionado
    pixel_value = impixel(img, x, y);  % Obtener el color del pixel

    % Almacenar el punto seleccionado como el vector
    vector = pixel_value;

% Crear una matriz de colores para asignar colores diferentes a cada clase
colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];

% Crear una figura para el gráfico 3D
figure;
grid on;
hold on;

% Plotear los puntos de cada clase con colores diferentes
for index = 1:n_classes
    clase = class_representatives{index};
    color = colores(index);
    plot3(clase(:, 1), clase(:, 2), clase(:, 3), 'o', 'MarkerSize', 10, 'MarkerFaceColor', color, 'MarkerEdgeColor', 'k');
end

% Plotear el vector seleccionado por el usuario con un color específico
plot3(vector(1), vector(2), vector(3), 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');


% Crear una leyenda para identificar cada clase
legend_str = cell(1, n_classes + 1);  % Inicializar celda de texto para leyenda
for index = 1:n_classes
    legend_str{index} = ['Clase ' num2str(index)];
end
legend_str{n_classes + 1} = 'Vector';
legend(legend_str);


% Configurar etiquetas para los ejes x, y, z
xlabel('Eje X');
ylabel('Eje Y');
zlabel('Eje Z');

% Establecer el título del gráfico
title('Gráfico 3D de Clases y Vector');

% Mostrar el gráfico
hold off;

    % Calcular las distancias de Mahalanobis utilizando la matriz "dato"
    distancias = zeros(1, n_classes);
    vector=vector';

    for index = 1:n_classes
        % Calcular la diferencia entre el vector y la media de la clase actual
        resta = vector - media_original{index};

        % Calcular la distancia de Mahalanobis utilizando la matriz "dato"
        distancia = (resta' * dato_inv{index}) * resta;

        % Almacenar la distancia en la matriz de distancias
        distancias(index) = distancia;
    end

    % Mostrar las distancias entre el vector y cada clase
    for index = 1:n_classes
        disp(['Distancia entre el vector y la Clase ' num2str(index) ': ' num2str(distancias(index))]);
    end

    % Determinar a qué clase pertenece el vector (la más cercana)
    [~, clase_pertenece] = min(distancias);

    % Determinar a qué clase pertenece el vector (la más cercana)
    [~, clase_pertenece] = min(distancias);

    if min(distancias) > 500
        disp('El vector no pertenece a ninguna clase');
    else
        disp(['El vector pertenece a la Clase ' num2str(clase_pertenece)]);
    end



    % Preguntar al usuario si desea probar otra vez
    prompt = '¿Desea probar otra vez el programa? (S/N): ';
    choice = input(prompt, 's');

    % Verificar si el usuario desea continuar o salir
    if ~ismember(upper(choice), {'S', 'N'})
        disp('Entrada no válida. Por favor, ingrese S o N.');
    elseif upper(choice) == 'N'
        break;  % Salir del bucle si el usuario ingresa 'N'
    end
end

disp('Programa terminado.');