%% === 4. REALIZACION DE TRANSFERENCIA DE APRENDIZAJE ===

%% == Preparacion de los datos de entrenamiento ==
load pathToImages
flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true);
flowernames = flwrds.Labels
% Las etiquetas necesarias para el entrenamiento pueden almacenarse en la
% propiedad 'labels' del almacen de datos de img. De manera predeterminada
% esta vacia.
%
% Se puede hacer que se determinen automaticamente
ds = iamgeDatastore(folder, 'IncludeSubfolders',true,'LabelSource', 'foldernames')
% Extraer propiedad 
flowernames = flwrds.Labels

%% == Dividir datos para entrenamiento (Training) y para pureba (Test) ==
%
% Dividir la base de datos
[ds1,ds2] = splitEachLabel(imds,p) % p es un valor de 0 a 1. Indica el porcentage en ds1.
% Mezclar de forma aleatoria los datos
[ds1,ds2] = splitEachLabel(imds,p,'randomized');
% Especificar el numero exacto de datos que toma de una etiqueta (bueno,
% malo)
[ds1,ds2] = splitEachLabel(imds,n) % n es el numero de archivos de cada categoria (etiqueta)

%% == Modificar las capas de uan red previamente entrenada ==
%
anet = alexnet;
layers = anet.Layers
%
% Crear una nueva capa totalmente coenctada
fclayer = fullyConnectedLayer(nNeuronas)
% Modificar elementos individuales mediante indexacion de arreglos estandar
% arreglos = vectores, matrices o hypermatrices
mylayers(nLayer) = mynewlayer
% Crear una nueva capa de salida para una red de clasificacion
cl = classificactionLayer
% Se puede hacer en un unico comando 
mylayers(nLayer) = classificationLayer

%% == Establecimiento de las opciones de entrenamiento ==
%
% Ver las opciones disponibles para un algoritmo de entrenamiento 
% específico 
opts = trainingOptions('sgdm');     % opciones predeterminadas para el algorimo sgdm
%       sgdm : descenso de gradiente estocastico con impetu
%                                       
% Ajustar el ratio de aprendizaje (u otros parametros que se visualizan en las opciones)
opts = trainingOptions('sgdm','Name',value)
opts = trainingOptions('sgdm', 'InitialLearnRate',0.001)

%% == Ejemplo de transferencia de aprendizaje ==
%
% Obtener las imagenes de entrenamiento
flower_ds = imageDatastore('Flowers','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(flower_ds,0.6);
numClasses = numel(categories(flower_ds.Labels));
%
% Crear una red mediante la modificacion de Alexnet
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;
%
% Establecer las opciones del algoritmo de entrenamiento
options = trainingOptions('sgdm','InitialLearnRate', 0.001);
%
% Realizar el entrenamiento
[flowernet,info] = trainNetwork(trainImgs, layers, options);
%
% Utilizar la red entrenada para clasificar imagenes de prueba
testpreds = classify(flowernet,testImgs);

%% == Evaluacion de una red tras el entrenamiento ==
%
% == Evalucaiccion del entrenamiento y rendimiento de la prueba ==
% Carga la informacion del entrenamiento
load pathToImages
load trainedFlowerNetwork flowernet info
% 
% Los campos TrainingLoss y TrainingAccuracy contienen un registro del 
% rendimiento de la red con los datos de entrenamiento en cada iteración.
% Grafica el training Loss
plot(info.TrainingLoss);  % ¿¿¿ FUNCION DE COSTE ???
% 
% Crea un almacen de datos 
dsflowers = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(dsflowers,0.98);
%
% Probar con los datos de prueba para ver el verdadero rendimiento
flwrPreds = classify(flowernet,testImgs)
%
% == Investigacion del rendimiento de la prueba ==
% 
load pathToImages.mat
pathToImages
flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(flwrds,0.98);
load trainedFlowerNetwork flwrPreds
% Comparar la clasificacion predicha con la conocida. Estas se almacenan en
% la propiedad Labels del alacen de datos
flwrActual = testImgs.Labels
% Determinar el numero de elementos de dos arreglos que coincidan
numCorrect = nnz(flwrPreds == flwrActual)
% Calcular la fraccion correcta
fracCorrect = numCorrect / length(flwrActual)
fracCorrect = numCorrect/numel(flwrPreds)
% Matriz de confusion
% es un recuento de cuántas imágenes de la clase j predijo la red que 
% estarían en la clase k.
confusionchart(knownclass,predictedclass)
confusionchart(testImgs.Labels, flwrPreds)




