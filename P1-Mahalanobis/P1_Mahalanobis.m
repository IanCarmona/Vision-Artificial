clase1 = [0 0 0; 1 0 0; 1 0 1; 1 1 0];
clase2 = [0 0 1; 0 1 1; 1 1 1; 0 1 0];

% Imprime la longitud de clase1 y clase2
disp("Longitud de clase1: " + length(clase1));
disp("Longitud de clase2: " + length(clase2));

claset1 = clase1' % Transponer para obtener una matriz 3x4
claset2 = clase2' % Transponer para obtener una matriz 3x4

media1 = (1/length(clase1)) * sum(clase1)'
media2 = (1/length(clase2)) * sum(clase2)'

covariaza1 = claset1 - media1
covariaza2 = claset2 - media2

c1t = covariaza1'
c2t = covariaza2'

a = (1/length(clase1))*(covariaza1*c1t)
b = (1/length(clase2))*(covariaza2*c2t)

ai = inv(a)
bi = inv(b)

vector = input("Ingrese el vector: ");

plot3(clase1(:,1),clase1(:,2),clase1(:,3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor','r')
grid on
hold on
plot3(clase2(:,1),clase2(:,2),clase2(:,3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor','g')
plot3(vector(:,1),vector(:,2),vector(:,3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor','b')

legend('clase1','clase2','vector')

x = vector(1,1);
y = vector(1,2);
z = vector(1,3);

if x <= 1 && y <= 1 && z <= 1

    disp("ESTA ADENTRO DEL CUBO");
    
    vector = vector'
    resta1 = vector - media1
    resta2 = vector - media2
    resta1t = resta1'
    resta2t = resta2'

    distancia1 = (resta1t * ai) * resta1;
    distancia2 = (resta2t * bi) * resta2;

    if distancia1 < distancia2
        disp("Distancia 1")
        disp(distancia1)
        disp("Distancia 2")
        disp(distancia2)
        disp("El punto pertenece a clase 1")
    else
        disp("Distancia 1")
        disp(distancia1)
        disp("Distancia 2")
        disp(distancia2)
        disp("El punto pertenece a clase 2")
    end

else
    disp("ESTA AFUERA DEL CUBO no tiene clase");
end