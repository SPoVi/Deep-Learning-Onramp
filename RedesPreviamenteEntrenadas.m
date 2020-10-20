%% === 2. USO DE REDES PREVIAMENTE ENTRENADAS ===
%% == Identificar objetos en alguna imagenes ==
%
% Importar imagenes
img = imread('filename.png');
% Mostar imagen en Matlab
imshow(img)

%% == Hacer una prediccion ==
%
% Carga de una red
net = alexnet % plugin de matlab
% Hacer una prediccion sobre una imagen
pred = classify(net,img)

%% == Arquitectura de una CNN ==
%
% Inspeccionar las capas de la red
ly = deepnet.Layers
% Inspeccionar una capa individual
inlayer = ly(1)     % primera capa (entrada)
outlayer = ly(end)  % ultima capa (salida)
% Tamaño de la capa de entrada
insz = inlayer.InputSize
% Nombres de las categorias que la red esta entrenada para predecir.
categorynames = outlayer.Classes

%% == Investigación de predicciones ==
%
% Obtener las puntuaciones predichas para todas las clases
[pred, scrs] = classify(net, img)
% Grafico de barras de la puntuaciones
bar(scrs)
% Threshold scores
highscores = scores > 0.1
% Grafico de barras con threshold
bar(scores(highscores))
% Añadir los nombres a las barras
xticklabels(categorynames(highscores))
% Distintos valores de umbral. Criterio dinamico
thresh = median(scores) + std(scores);
highscores = scores > thresh;








