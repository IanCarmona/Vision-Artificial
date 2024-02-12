%% Práctica 3 - Perceptron

% Integrantes:
% Carmona Serrano Ian Carlo
% Méndez López Luz Fernanda
% Rojas Alarcon Sergio Ulises

% Limpiando el entorno de trabajo
clc;
clear;
close all;
warning off all;

% Limpiando el entorno de trabajo
clc;
clear;
close all;
warning off all;

while true
    % Solicitar al usuario ingresar los pesos y el coeficiente de error
    w = input('Ingrese los cuatro pesos como un arreglo [wx, wy, wz, w0]: ');
    
    r = 0; % Inicializa r con un valor inválido para entrar en el bucle
    
    while r <= 0
        % Solicitar al usuario ingresar el valor del coeficiente de error r
        r = input('\nIngrese el valor del coeficiente de error r (debe ser mayor a 0): ');
        if r <= 0
            disp('El coeficiente de error debe ser mayor a 0.');
        end
    end

    % Definición de los datos de entrada X y las etiquetas y (las clases)
    X = [0 0 0; 1 0 0; 1 1 0; 1 0 1; 0 1 0; 0 1 1; 0 0 1; 1 1 1];
    y = [0 0 0 0 1 1 1 1];

    % Inicialización de variables de control
    converge = false;
    etapa = 0;

    % Bucle principal para el aprendizaje del perceptrón
    % Bucle que se ejecutará hasta que converge sea true
    while ~converge
        converge = true;
        disp(' ');
        disp(['Etapa ' num2str(etapa + 1) ':']);

        for i = 1:length(X)
            xn = [X(i, :) 1];
            fsal = dot(xn, w);

            % Verificar si la salida del perceptrón coincide con la etiqueta
            if fsal >= 0 && y(i) == 0
                disp(['Entrada: ' mat2str(X(i, :)) ', Clase: 1']);
                w = w - r * xn; % Corrección para clase 1
                converge = false;
            elseif fsal <= 0 && y(i) == 1
                disp(['Entrada: ' mat2str(X(i, :)) ', Clase: 2']);
                w = w + r * xn; % Corrección para clase 2
                converge = false;
            else
                disp(['Entrada: ' mat2str(X(i, :)) ', Sin cambios']);
            end
        end
        etapa = etapa + 1;
    end

    % Mostrar los valores finales de los pesos
    disp(' ');
    disp(['Valores finales de los pesos: ' mat2str(w)]);

    % Calcula la ecuación del plano
    plane_equation = sprintf('%.2f*x + %.2f*y + %.2f*z + %.2f = 0', w(1), w(2), w(3), w(4));

    % Crear una nueva figura para el gráfico
    figure;

    % Crear el eje
    axis = axes('Parent', gcf);
    hold on;
    grid on;

    % Dibujar los puntos de datos
    for i = 1:length(X)
        if y(i) == 0
            plot3(axis, X(i, 1), X(i, 2), X(i, 3), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
        else
            plot3(axis, X(i, 1), X(i, 2), X(i, 3), 'kh', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        end
    end

    xlabel('X');
    ylabel('Y');
    zlabel('Z');

    % Crear una malla para el plano de decisión
    [xx, yy] = meshgrid(linspace(0, 1, 2), linspace(0, 1, 2));
    zz = (-w(1) * xx - w(2) * yy - w(4)) / w(3);

    % Establecer el color del plano de decisión como morado
    surf(axis, xx, yy, zz, 'FaceAlpha', 0.5, 'FaceColor', 'm');

    % Establecer el título del gráfico con la ecuación del plano
    title(axis, ['Plano: ' plane_equation]);

    % Etiquetar los puntos de datos y clases
    text(0, 0, 0, 'Clase 1', 'Color', 'b');
    text(1, 0, 0, 'Clase 1', 'Color', 'b');
    text(1, 1, 0, 'Clase 1', 'Color', 'b');
    text(1, 0, 1, 'Clase 1', 'Color', 'b');
    text(0, 1, 0, 'Clase 2', 'Color', 'r');
    text(0, 1, 1, 'Clase 2', 'Color', 'r');
    text(0, 0, 1, 'Clase 2', 'Color', 'r');
    text(1, 1, 1, 'Clase 2', 'Color', 'r');

    hold off;
    
    % Preguntar al usuario si desea realizar nuevos cálculos
    respuesta = input('\n¿Desea realizar nuevos cálculos? (S/N): ', 's');
    
    % Si la respuesta no es 'S' o 's', terminar el programa
    if ~(respuesta == 'S' || respuesta == 's')
        break;
    end
end


