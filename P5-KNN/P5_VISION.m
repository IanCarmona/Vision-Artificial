%% Práctica 5 - KNN

% Integrantes:
% Carmona Serrano Ian Carlo
% Mendez Lopez Luz Fernanda
% Rojas Alarcon Sergio Ulises

clc;
clear;
close all;
warning off all;

fprintf("Menú de Imágenes: \n1. Desierto \n2. Montaña \n3. Montaña Blanca y Arena Rosa \n4. Playa\n\n");
ImgChoice = input('Escoge la imagen con la que deseas trabajar: ');

if ImgChoice==1
    Img = imread('Desert.jpeg');
elseif ImgChoice==2
    Img = imread('Mountain.jpg');
elseif ImgChoice==3
    Img = imread('Landscape.jpg');
elseif ImgChoice==4
    Img = imread('Beach.jpg');
end

[rows, cols, ~] = size(Img);


colores = ['y', 'm','r', 'c','g', 'w', 'k','b'];

NumClasses = input('Número de clases: ');

NumRep = input('Número de representantes: ');

Total_elem = NumRep * NumClasses;
classes_elements = zeros(Total_elem, 3);


imshow(Img);
hold on;

centroide = zeros(length(NumClasses), 2);
dataset_rgb = zeros(Total_elem, 3);
dataset_labels = zeros(Total_elem, 1); %-> Clases
index = 1;

dataset_rgb = [-2 0; -1 -2; 1 0; 2 0]; 
dataset_labels = [0 0 1 1];


user_input = 's';

pixel=input('ingrese el vector: ');

num_puntos=1;

while strcmp(user_input, 's')
    for i = 1:num_puntos
        % Espera a que el usuario haga clic en la imagen
        [x, y] = ginput(1);

        % Redondea las coordenadas a números enteros
        x = round(x);
        y = round(y);

        % Obtiene los valores RGB del punto seleccionado
        rgb_values = Img(y, x, :);

        % Almacena los valores RGB en el arreglo
        pixel(i, :) = reshape(rgb_values, 1, 3);

        % Muestra los valores RGB
        fprintf('Valores RGB del punto %d en (x, y): (%d, %d, %d)\n', i, rgb_values(1), rgb_values(2), rgb_values(3));

        % Agrega un punto negro en las coordenadas (x, y)
        hold on;
        plot(x, y, 'k.', 'MarkerSize', 10); % 'k.' representa un punto negro
    end

    k = input('Ingrese el valor de k (vecinos): ');

    fprintf("\nDistancias:")
    fprintf("\n1. Dist. Euclidiana \n2. Mahalanobis\n\n");
    choice = input("Selecciona una opción: ");

    if choice == 1
        disp("\nDistancia Euclidiana");

        % Inicializa un vector para almacenar todas las distancias
        todas_distancias = zeros(Total_elem, 1);

        % Calcula la distancia euclidiana entre el punto nuevo y todos los puntos de todas las clases
        for i = 1:Total_elem
            todas_distancias(i) = sqrt(sum((dataset_rgb(i, :) - pixel).^2));
        end

        % Encuentra las k distancias menores y sus clases correspondientes
        [distancias_ordenadas, clases_predichas] = mink(todas_distancias, k);

        disp('Las k distancias menores y sus clases correspondientes son:');
        for i = 1:k
            fprintf('Distancia %d: %.2f, Clase: %d\n', i, distancias_ordenadas(i), dataset_labels(clases_predichas(i)));
        end

        % Encuentra la clase más común entre las k clases cercanas
        clase_predicha = mode(dataset_labels(clases_predichas(1:k)));

        fprintf('\nEl punto pertenece a la clase %d\n', clase_predicha);

    elseif choice == 2
        disp("Mahalanobis");
        % Inicializa un vector para almacenar todas las distancias
        % Mahalanobis
    
        % Calcula la matriz de covarianza de los datos
        cov_matrix = cov(dataset_rgb);
        inv_cot_var = inv(cov_matrix);
        % Inicializa un vector para almacenar todas las distancias
        todas_distancias = zeros(Total_elem, 1);

        % Calcula la distancia euclidiana entre el punto nuevo y todos los puntos de todas las clases
        for i = 1:Total_elem
            todas_distancias(i) = Mahalanobis(pixel, dataset_rgb(i,:), inv_cot_var);
        end

        % Encuentra las k distancias menores y sus clases correspondientes
        [distancias_ordenadas, clases_predichas] = mink(todas_distancias, k);

        disp('Las k distancias menores y sus clases correspondientes son:');
        for i = 1:k
            fprintf('Distancia %d: %.2f, Clase: %d\n', i, distancias_ordenadas(i), dataset_labels(clases_predichas(i)));
        end

        % Encuentra la clase más común entre las k clases cercanas
        clase_predicha = mode(dataset_labels(clases_predichas(1:k)));

        fprintf('\nEl punto pertenece a la clase %d\n', clase_predicha);

    end



    disp('¿Deseas usar otro metodo de distancia? s: Continuar. Cualquier otra tecla: Salir');
    user_input = input('Teclea la opción deseada: ', 's');
end


function point_in_image = point_is_in_image(x, y, img_size_x, img_size_y)
    if  (x >= 1 && y >= 1) && (x <= img_size_x && y <= img_size_y)
        point_in_image = true;
        return
    end

    point_in_image = false;
end

function [dataset_rgb_values, dataset_labels] = get_rgb_from_coordinates(image, class_x_values, class_y_values, elements_p_class, class_no)
    dataset_rgb_values = zeros(elements_p_class, 3);
    dataset_labels = zeros(elements_p_class, 1);
    for i = 1 : elements_p_class
        rgb_value = image(class_y_values(i), class_x_values(i), :);
        dataset_rgb_values(i, :) = rgb_value;
        dataset_labels(i) = class_no;
    end
end


function [x_coordinates, y_coordinates] = get_n_points_inside_image_limits(c_grav_x, c_grav_y, img_size_x, img_size_y, elements_p_class)
    separated_factor = 30;
    x_coordinates = int32(randn(1, elements_p_class) .* separated_factor + c_grav_x);
    y_coordinates = int32(randn(1, elements_p_class) .* separated_factor + c_grav_y);

    for i = 1 : elements_p_class
        x_value = x_coordinates(i);
        if x_value < 1
            x_value = 1;
        elseif x_value > img_size_x
            x_value = img_size_x;
        end

        y_value = y_coordinates(i);
        if y_value < 1
            y_value = 1;
        elseif y_value > img_size_y
            y_value = img_size_y;
        end

        x_coordinates(i) = x_value;
        y_coordinates(i) = y_value;
    end
end

function Distancia = Mahalanobis(ValDesc, puntoClase, InvMatCov)
    Distancia = (ValDesc - puntoClase) * InvMatCov * (ValDesc - puntoClase)';
end




