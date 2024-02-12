%% Práctica 4 - Clasificación

% Integrantes:
% Carmona Serrano Ian Carlo
% Mendez Lopez Luz Fernanda
% Rojas Alarcon Sergio Ulises

% Limpiando el entorno de trabajo
clc;
clear;
close all;
warning off all;

% Ingresar el numero de clases y de representantes
NumClasses = input('Número de clases: ');
NumRep = input('Número de representantes: ');

% Calcular total de elementos
Total_elem = NumRep * NumClasses;

% Leer imagen y obtener dimensiones
Img = imread('Desert.jpeg');
[rows, cols, ~] = size(Img);

% Mostrar la imagen
imshow(Img);
hold on;

% Almacenar centroides, RGB y etiquetas
centroide = zeros(length(NumClasses), 2);
dataset_rgb = zeros(Total_elem, 3);
dataset_labels = zeros(Total_elem, 1); 

% Definiendo una lista de colores para los puntos
colores = ['y', 'm','r', 'c','g', 'w', 'k','b'];

index = 1;


while index <= NumClasses
    % Hacer click en la imagen para selecc. clases y obtener
    % coordenadas
    centroide(index, :) = ginput(1);
    centroide_x = centroide(index, 1);
    centroide_y = centroide(index,2);

    %fprintf('\nCapturado el centroide de clase %d:\n', index);
    
    % Validación para tomar puntos dentro de la imagen
    if ~point_is_in_image(centroide_x, centroide_y, cols, rows)
        fprintf('\n\nPor favor selecciona solo puntos dentro de la imagen\n');
    else
        % Plotear el centroide si está dentro de la imagen
        plot(centroide_x, centroide_y, 'o', 'MarkerSize', 5, 'MarkerFaceColor', 'black');
        
        % Obtener las coordenadas de los puntos cercanos al centroide
        [x_coordinates, y_coordinates] = get_n_points_inside_image_limits(centroide_x, centroide_y, cols, rows, NumRep);
        
        % Obtener valores RGB de los puntos cercanos
        [class_rgb_values, class_labels] = get_rgb_from_coordinates(Img, x_coordinates, y_coordinates, NumRep, index);
        
        % Calculo de índices de inicio y fin para valores RGB
        start_idx = (index - 1) * NumRep + 1;
        end_idx = start_idx + NumRep - 1;

        % Almacenar los valores RGB en el dataset
        dataset_rgb(start_idx:end_idx, :) = class_rgb_values;

        % Almacenar etiquetas en el dataset
        dataset_labels(start_idx:end_idx) = class_labels;
    
        % Generar los representantes de las clases
        color = colores(index);
        scatter(x_coordinates, y_coordinates, color, "filled");
        
        index = index + 1;
    end
end

while true
    fprintf("\nTipos de clasificación:")
    fprintf("\n1. Dist. Euclidiana \n2. Mahalanobis \n3. Máxima Probabilidad\n\n");
    choice = input("Selecciona una opción: ");
    
    k_value_knn = -1;

    if choice==1
        % Llamada a función de distancia euclidiana
        selected_classifier = @distancia_euclidiana;
        classifier_name = "Distancia Euclidiana";
    elseif choice == 2
        % Llamada a función de Mahalanobis
        selected_classifier = @distancia_mahalannobis;
        classifier_name = "Mahalanobis";   
    elseif choice == 3
        % Llamada a a función de Max. Prob. 
        selected_classifier = @max_prob;
        classifier_name = "Criterio de Máxima probabilidad";
    
    end
    
    fprintf("\n\nOpción Seleccionada: %s\n\n", classifier_name);

    %RESUSTITUCIÓN %%
    %disp("Para RESUSTITUCIÓN: ")
    %disp(' ');
    total_datos_entrenamiento = Total_elem;
    resustitution_conf_matrix = obtener_matriz_conf(selected_classifier, NumClasses, dataset_rgb, dataset_labels, total_datos_entrenamiento, k_value_knn);
    %disp("Matriz de Confusión")
    %disp(resustitution_conf_matrix);
    resustitution_accuracy = obtener_eficiencia(resustitution_conf_matrix);
     y_bar = diag((resustitution_conf_matrix * 100 / NumRep));
     x_bar = 1:length(y_bar);
     figure(4);
    bar(x_bar, y_bar, "magenta");
    hold on;
    title("Resustitución");


    %CROSS-VALIDATION, 20 iteraciones %%
    iterations = 20;
    %fprintf("Para CROSS-VALIDATION:\n")
    %disp(' ');
    total_datos_entrenamiento = floor(Total_elem / 2);
    cross_val_global_conf_matrix = zeros(NumClasses, NumClasses);
    for i = 1 : iterations
        cross_val_conf_matrix = obtener_matriz_conf(selected_classifier, NumClasses, dataset_rgb, dataset_labels, total_datos_entrenamiento, k_value_knn);
        %fprintf("(%d): \n", i);
        %disp(cross_val_conf_matrix);
        cross_val_global_conf_matrix = cross_val_global_conf_matrix + cross_val_conf_matrix;
    end
    cross_val_accuracy = obtener_eficiencia(cross_val_global_conf_matrix);

    %Grafica
    y_bar = diag((cross_val_global_conf_matrix * 100 / NumRep));
    x_bar = 1:length(y_bar);
    figure(5);
    bar(x_bar, y_bar, "cyan");
    hold on;
    title("Cross Validation");

        
    %LEAVE ONE OUT
    %disp("Para LEAVE ONE OUT:")
    %disp(' ');
    leave_one_out_conf_matrix = leave_one_out_using_f(selected_classifier, NumClasses, dataset_rgb, dataset_labels, k_value_knn);
    %disp("Matriz de Confusión")
    %disp(leave_one_out_conf_matrix);
    leave_one_out_accuracy = obtener_eficiencia(leave_one_out_conf_matrix);

    resustitution_mean=resustitution_accuracy*100; 
    loo_mean=leave_one_out_accuracy*100; 
    cross_v_mean=cross_val_accuracy*100; 
    
    fprintf("Eficacia de %s:\n\n", classifier_name);
    fprintf("Resustitución: %f (%f %%)\n", resustitution_accuracy,resustitution_mean);
    fprintf("Cross-Validation: %f (%f %%)\n", cross_val_accuracy,cross_v_mean);
    fprintf("Leave One Out: %f (%f %%)\n", leave_one_out_accuracy,loo_mean);

     %Grafica
    y_bar = diag((leave_one_out_conf_matrix * 100 / NumRep));
    x_bar = 1:length(y_bar);
    figure(6);
    bar(x_bar, y_bar, "yellow");
    hold on;
    title("Leave One Out");

%     x_acc = ["Restitucion", "CrossValidation", "Leave One Out"];
%     y_acc = [resustitution_accuracy,cross_val_accuracy,leave_one_out_accuracy];
%     figure(8);
%     bar(x_acc, y_acc, "yellow");
%     hold on;
%     title("Mejor Clasificador: ")

    % Preguntar al usuario si desea realizar nuevos cálculos
    respuesta = input('\n¿Desea realizar nuevos cálculos? (S/N): ', 's');
    
    if ~(respuesta == 'S' || respuesta == 's')
        break; % Si la respuesta no es 'S' o 's', termina el programa
    end
end

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% FUNCIONES
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function accuracy = obtener_eficiencia(conf_matrix)
    [rows, cols] = size(conf_matrix);
    total_predictions = 0;
    true_positives = 0;

    for i = 1 : rows
        for j = 1 : cols
            predictions_count = conf_matrix(i, j);
            total_predictions = total_predictions + predictions_count;
            if i == j
                true_positives = true_positives + predictions_count;
            end
        end
    end

    accuracy = true_positives / total_predictions;
end

function conf_matrix = obtener_matriz_conf(selected_criteria_function, no_classes, X, y, total_train_elements, k_for_knn)
    [total_elements_count, ~] = size(y);
    if total_train_elements == total_elements_count
        train_data = X;
        test_data = X;
        train_labels = y;
        test_labels = y;
    else
        [train_data, train_labels, test_data, test_labels] = get_test_train_data(X, y, total_train_elements);
    end

    [test_elements_count, ~] = size(test_labels);
    conf_matrix = zeros(no_classes, no_classes);
    
    for element_no = 1 : test_elements_count
        vector_x = test_data(element_no, :);
        expected_output = test_labels(element_no);

        predicted_class = -1;
        
        if k_for_knn <= 0
            [predicted_class, ~] = selected_criteria_function(train_data, train_labels, no_classes, vector_x);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_for_knn, vector_x);
        end

        conf_matrix(expected_output, predicted_class) = conf_matrix(expected_output, predicted_class) + 1;
    end
end

function conf_matrix = leave_one_out_using_f(selected_criteria_function, no_classes, X, y, k_for_knn)
    [total_elements_count, ~] = size(y);
    conf_matrix = zeros(no_classes, no_classes);

    for element_no = 1:total_elements_count
        % Use all data points except the one at "element_no" for training
        train_data = X;
        train_data(element_no, :) = [];
        
        train_labels = y;
        train_labels(element_no, :) = [];
        
        % Use the data point at "element_no" for testing
        test_data = X(element_no, :);
        test_labels = y(element_no);

        predicted_class = -1;
        
        if k_for_knn <= 0
            [predicted_class, ~] = selected_criteria_function(train_data, train_labels, no_classes, test_data);
        else
            predicted_class = knn_euclidean(train_data, train_labels, k_for_knn, test_data);
        end
        
        conf_matrix(test_labels, predicted_class) = conf_matrix(test_labels, predicted_class) + 1;
    end
end


function [class_no] = knn_euclidean(dataset_rgb, dataset_labels, k, vector)
    distances = zeros(size(dataset_rgb, 1), 1);
    
    for i = 1:size(dataset_rgb, 1)
        distances(i) = norm(vector - dataset_rgb(i, :));
    end
    
    [~, sortedIndices] = sort(distances);
    nearestNeighbors = sortedIndices(1:k);
    
    neighborLabels = dataset_labels(nearestNeighbors);
    
    class_no = mode(neighborLabels);
end

function [class_no, min_distance] = distancia_euclidiana(dataset_rgb, dataset_labels, classes_count, vector)
    min_distance = inf;
    class_no = -1;
    
    for class_index = 1:classes_count
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
        
        mu = mean(class_values);
        
        distance = norm(vector - mu);
        
        if distance < min_distance
            min_distance = distance;
            class_no = class_index;
        end
    end
end

function [class_no, max_likelihood] = max_prob(dataset_rgb, dataset_labels, classes_count, vector)
    max_likelihood = -inf;
    class_no = -1;
    
    for class_index = 1:classes_count
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
        
        mu = mean(class_values);
        Sigma = cov(class_values);
        
        k = size(vector, 2); 
        delta = vector - mu;
        likelihood = (1 / ((2*pi)^(k/2) * sqrt(det(Sigma)))) * exp(-0.5 * delta * inv(Sigma) * delta');
        
        if likelihood > max_likelihood
            max_likelihood = likelihood;
            class_no = class_index;
        end
    end
end

function [class_no, current_min] = distancia_mahalannobis(dataset_rgb, dataset_labels, classes_count, vector)
    current_min = inf;
    class_no = -1;
    
    for class_index = 1:classes_count
        class_values = get_class_values(dataset_rgb, dataset_labels, class_index);
    
        mu = mean(class_values);
        
        Sigma = cov(class_values);
        
        Sigma_inv = inv(Sigma);
        
        delta = vector - mu;
        D2 = delta * Sigma_inv * delta';
        
        dist = abs(D2);

   
        if dist < current_min
            current_min = dist;
            class_no = class_index;
        end
        
    end
end

function class_values = get_class_values(dataset_values, dataset_labels, desired_class)
    [rows_count, ~] = size(dataset_labels);    
    class_values = [];
    for i = 1 : rows_count
        label = dataset_labels(i);
        if label == desired_class
            class_values = [class_values; dataset_values(i, :)];
        end
    end
end


function [train_data, train_labels, test_data, test_labels] = get_test_train_data(dataset, labels, total_train_elements)
   
    classes = unique(labels);

    train_data = [];
    train_labels = [];
    test_data = [];
    test_labels = [];
    
    n_classes = length(classes);
    train_elements_per_class = floor(total_train_elements / n_classes);

    remainder = mod(total_train_elements, n_classes);

    for i = 1:n_classes
        class_indices = find(labels == classes(i));
        class_data = dataset(class_indices, :);
        class_labels = labels(class_indices);
        
        idx = randperm(length(class_indices));
        class_data = class_data(idx, :);
        class_labels = class_labels(idx);
    
        if total_train_elements < n_classes
            if i <= total_train_elements
                n_take = 1;
            else
                n_take = 0;
            end
        else
            if i <= remainder
                n_take = train_elements_per_class + 1;
            else
                n_take = train_elements_per_class;
            end
        end

        class_train_data = class_data(1:n_take, :);
        class_train_labels = class_labels(1:n_take);
        class_test_data = class_data(n_take+1:end, :);
        class_test_labels = class_labels(n_take+1:end);

        train_data = [train_data; class_train_data];
        train_labels = [train_labels; class_train_labels];
        test_data = [test_data; class_test_data];
        test_labels = [test_labels; class_test_labels];
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

function [dataset_rgb_values, dataset_labels] = get_rgb_from_coordinates(image, class_x_values, class_y_values, elements_p_class, class_no)
    dataset_rgb_values = zeros(elements_p_class, 3);
    dataset_labels = zeros(elements_p_class, 1);
    for i = 1 : elements_p_class
        rgb_value = image(class_y_values(i), class_x_values(i), :);
        dataset_rgb_values(i, :) = rgb_value;
        dataset_labels(i) = class_no;
    end
end

function point_in_image = point_is_in_image(x, y, img_size_x, img_size_y)
    if  (x >= 1 && y >= 1) && (x <= img_size_x && y <= img_size_y)
        point_in_image = true;
        return
    end

    point_in_image = false;
end