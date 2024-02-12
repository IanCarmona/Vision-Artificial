% Integrantes:
% Rojas Alarcon Sergio Ulises

% Limpiando el entorno de trabajo
clc;
clear;
close all;
warning off all;

%% Despues de programar el algoritmo se llega a:
% fsal=2x1+2x2-1

% Graficando las clases
c1=[0 0]
c2=[0 1 1; 1 0 1]

x=[0,0.5]
y=[0.5,0]

figure(1)
plot(c1(1,1),c1(1,2),'ro','MarkerSize',10,'MarkerFaceColor','r')
grid on
hold on
plot(c2(1,:),c2(2,:),'go','MarkerSize',10,'MarkerFaceColor','g')
plot(x,y,'k')

disp('Fin del Programa')

