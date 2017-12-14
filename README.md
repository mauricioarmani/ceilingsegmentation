# Segmentação de Teto para Localização Robôs

## Pastas
data/ - Pasta que contém dados de treino e validação

data_maur/ - Pasta contém dados que não foram utilizados por inconsistência na anotação

data_test/ - Pasta utilizada para rodar dados de teste (anotações podem ser desprezadas)

data_test2/ - Mesma coisa

logs/ - Pasta onde é salvo o modelo treinado em diversos checkpoints. Também onde ficam salvas as imagens das predições, ground_truth e a imagem de entrada.

model/ - Pasta onde carrega os pesos da FCN-VGG19.

## Modificações da implementação original FCN-VGG19
Implementação original: https://github.com/shekkizh/FCN.tensorflow:

data_selection.py - criado para selecionar estocasticamente as imagens de treino.

FCN.py- utilizado para treinar, validar e testar:

* $python FCN.py (TREINO)
* $python FCN.py --mode visualize (PREDIÇÃO)

name_correction.py - script criado para ajustar o nome dos dados

read_MITSceneParsingData - Modificado a forma de aquisição dos dados.

## Dependencias:

`pip install python numpy tensorflow-gpu`

## Predição
1 - Criar pasta model/ na raiz e colocar dentro o grafo da VGG19: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat.

2 - No arquivo predict.py colocar o caminho da imagem em IMAGE_PATH

3 - Rodar o arquivo predict.py

4 - Resultado será salvo na pasta logs/
