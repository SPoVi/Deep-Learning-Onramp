%% === 3. GESTIÓN DE COLECCIONES DE DATOS DE IMAGEN ===
%
%% == Almacenes de datos de imagenes ==
%
% Crear almacen de datos
imds = imageDatastore('file*.jpg'); % * para especificar varios archivos
% Extraer el nombre de los archivos
fname = imds.Files;
% Leer una imagen (numeros)
I = readimage(ds,n) % ds: dataStore , n: posicion en el ds
        % read : importa la imagenes una por una
        % readimage: importa una única imagen especifica
        % readall: importa todas las imagenes a una unica celda de una
        % matriz (cada img en una celda independiente)
% Classificar imagenes de almacen de datos
preds = classify(net, imds);
[preds, scores] = classify(net, imds);
% Maximo de cada imagen
max(scores,[],2)
% Tambien se puede hacer lo de threshold y barras

%% == Preparacion de imagenes para utilizarlas como entrada ==
%
% Ver el tamaño de la img
sz = size(img)
% Ver el tamaño de la img que requiere la red
net = alexnet;
inlayer = net.Layers(1);    %input layer
insz = inlyer.InputSize;
% Redimensionar una img
imgresz = imresize(img,[numrows numcols]);

%% == Procesamiento de imagenes en un almacen de datos ==
%
% Creacion de almacen de datos
imds = imageDatastore('*.jpg')
% Los almacenes de datos de imágenes aumentadas pueden realizar 
% preprocesamientos simples de una colección completa de imágenes
auds = augmentedImageDatastore([r c],imds)
% Clasificacion datastore
preds = classify(net, auds)

%% == Procesamiento del color con almacenes de datos de imagenes ==
%
imds = imageDatastore('file*.jpg');
% Mostrar todas las imagenes
montage(imds);
% Arreglo tridimensional
auds = augmentedImageDatastore([n m], imds, 'ColorPreprocessing', 'gray2rgb');
% Clasificacion
preds = classify(net, auds);

%% == Crear un almacen de datos mediante subcarpetas ==
%
% Incluir subcarpetas en la busqueda
flwrds = imageDatastore('folder', 'IncludeSubfolders', true);
% Clasificar
preds = classify(net,flwrds);