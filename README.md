# Transfer Learning para Clasificación de Frutas y Verduras

**Proyecto 3 - Inteligencia Artificial**  
**INFO1185** | **Prof. Dr. Ricardo Soto Catalán** | **Diciembre 2025**

## Descripción del Proyecto

Este proyecto implementa **Transfer Learning** usando PyTorch para clasificar imágenes de frutas y verduras. Se desarrollan y comparan dos variantes de clasificadores basados en EfficientNetV2-S preentrenado, siguiendo los requerimientos específicos de la rúbrica del curso.

### Objetivos

- Aplicar Transfer Learning con EfficientNetV2-S preentrenado
- Implementar dos variantes del clasificador (simple y embudo)
- Comparar el impacto de técnicas de regularización (Batch Normalization y Dropout)
- Evaluar modelos con métricas completas y visualizaciones
- Analizar resultados según criterios de la rúbrica

## Arquitectura del Proyecto

### Versión 1: Clasificador Simple

- **Estructura**: EfficientNetV2-S + 1 capa Linear
- **Características**: Sin Batch Normalization ni Dropout
- **Objetivo**: Baseline simple para comparación

### Versión 2: Clasificador Embudo

- **Estructura**: EfficientNetV2-S + arquitectura tipo embudo (512 → 256 → 128 → num_classes)
- **Variantes**:
  - Sin regularización (sin BN ni Dropout)
  - Con regularización completa (BN + Dropout)
- **Objetivo**: Evaluar impacto de regularización

## Estructura del Proyecto

```
ia_proyecto_3/
├── modelo_v1.ipynb                    # Implementación Versión 1
├── modelo_v2.ipynb                    # Implementación Versión 2 (ambas variantes)
├── utils/                             # Módulos de utilidad
│   ├── data_proccess.py              # Carga y procesamiento de datos
│   ├── train_model.py                # Entrenamiento de modelos
│   ├── evaluate_model.py             # Evaluación de modelos
│   ├── predict_images.py             # Predicciones individuales
│   └── select_widget_imagen.py       # Widget interactivo
├── datos/                             # Dataset organizado
│   ├── train/                        # Imágenes de entrenamiento
│   ├── validation/                   # Imágenes de validación
│   └── test/                         # Imágenes de prueba
├── README.md                          # Este archivo
└── *.pth                             # Modelos entrenados guardados
```

## Instalación y Configuración

### 1. Requisitos del Sistema

```bash
# Python 3.7+
# CUDA opcional (recomendado para GPU)
```

### 2. Instalación de Dependencias

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pillow numpy ipywidgets
```

### 3. Configuración del Dataset

El dataset debe organizarse en la siguiente estructura:

```
datos/
├── train/
│   ├── clase1/
│   ├── clase2/
│   └── ...
├── validation/
│   ├── clase1/
│   ├── clase2/
│   └── ...
└── test/
    ├── clase1/
    ├── clase2/
    └── ...
```

## Uso del Proyecto

### Opción 1: Ejecutar Modelo V1

1. Abrir `modelo_v1.ipynb` en Jupyter Notebook
2. Ejecutar todas las celdas secuencialmente
3. El modelo entrenado se guardará como `efficientnetv2_s_v1.pth`

### Opción 2: Ejecutar Modelo V2

1. Abrir `modelo_v2.ipynb` en Jupyter Notebook
2. Ejecutar todas las celdas secuencialmente
3. Compara automáticamente ambas variantes (sin y con regularización)

### Configuración de Parámetros

Los principales parámetros están definidos en cada notebook:

```python
# Configuración de datos
data_dir = "datos"
img_size = 224
batch_size = 32

# Configuración de entrenamiento
learning_rate = 0.001
num_epochs = 50 (V2) / 10 (V1)
patience = 5
```

## Modelos Implementados

### Modelo V1: EfficientNetV2-S Simple

```python
# Backbone: EfficientNetV2-S preentrenado
# Clasificador: Linear(in_features → num_classes)
# Sin regularización
model.classifier = nn.Linear(in_features, num_classes)
```

### Modelo V2 Sin Regularización

```python
# Arquitectura embudo:
nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, num_classes)
)
```

### Modelo V2 Con Regularización

```python
# Arquitectura embudo + BN + Dropout:
nn.Sequential(
    nn.Linear(num_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    # ... (repetir patrón)
    nn.Linear(128, num_classes)
)
```

## Funcionalidades Implementadas

### Módulo data_proccess.py

- `create_transforms()`: Transformaciones de datos con data augmentation
- `load_datasets()`: Carga de datasets con división train/val/test
- `create_dataloaders()`: Creación de DataLoaders optimizados

### Módulo train_model.py

- `train_model()`: Función completa de entrenamiento con early stopping
- `plot_training_metrics()`: Visualización de curvas de entrenamiento

### Módulo evaluate_model.py

- `evaluate_model()`: Evaluación completa en conjunto de test
- Generación automática de matriz de confusión y métricas por clase

### Módulo predict_images.py

- `predict_random_from_test()`: Predicciones aleatorias del conjunto de test
- `predict_single_image()`: Predicción de imagen individual
- Visualización de resultados con confianza

### Módulo select_widget_imagen.py

- Widget interactivo para selección manual de imágenes
- Predicción en tiempo real con visualización

## Resultados y Evaluación

### Métricas Generadas

- **Precisión (Accuracy)** por clase y global
- **Precisión (Precision)** y **Sensibilidad (Recall)** por clase
- **Matriz de confusión** con visualización heatmap
- **Curvas de pérdida** durante entrenamiento
- **F1-Score** y reporte de clasificación completo

### Visualizaciones Automáticas

- Curvas de entrenamiento (pérdida y precisión)
- Matriz de confusión por modelo
- Comparación entre arquitecturas
- Muestras de predicciones con confianza

### Análisis Implementado

- Comparación cuantitativa entre V1 y V2
- Impacto de regularización (BN + Dropout)
- Análisis de estabilidad del entrenamiento
- Gap entre entrenamiento y validación
- Tiempo de convergencia por modelo

## Características Técnicas

### Transfer Learning

- Backbone: EfficientNetV2-S preentrenado en ImageNet
- Features congeladas durante entrenamiento
- Solo clasificador personalizado entrenable
- Aprovecha representaciones preaprendidas

### Técnicas de Entrenamiento

- Early stopping con patience configurable
- Optimizador Adam con learning rate 0.001
- CrossEntropyLoss como función de pérdida
- Data augmentation solo en conjunto de entrenamiento

### Regularización (V2 Con)

- BatchNormalization antes de ReLU
- Dropout con rate 0.3 entre capas
- Prevención de overfitting
- Mejora en generalización

## Solución de Problemas

### Error: Dataset no encontrado

```python
# Verificar estructura de carpetas en 'datos/'
# Asegurar que existen train/, validation/, test/
```

### Error: Sin GPU disponible

```python
# El código detecta automáticamente CPU/GPU
# Funciona en ambos pero GPU es más rápido
```

### Error: Memoria insuficiente

```python
# Reducir batch_size de 32 a 16 o 8
batch_size = 16
```

### Error: Widget no funciona

```python
# Instalar ipywidgets:
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## Para la Presentación

### Puntos Clave Implementados

1. **Problema**: Clasificación multi-clase con Transfer Learning
2. **Dataset**: Estructura train/val/test con transformaciones
3. **Modelos**: 3 variantes implementadas y entrenadas
4. **Entrenamiento**: Early stopping, métricas automáticas
5. **Evaluación**: Completa con matrices y métricas por clase
6. **Comparación**: Análisis automático entre arquitecturas
7. **Predicción**: Sistema interactivo y aleatorio

### Resultados Disponibles

- Accuracy final por modelo
- Curvas de convergencia
- Matrices de confusión
- Tiempo de entrenamiento
- Gap train-validation
- Mejor época por modelo

## Información del Proyecto

**Modelo utilizado**: EfficientNetV2-S  
**Framework**: PyTorch  
**Técnica**: Transfer Learning  
**Evaluación**: 3 arquitecturas completas

## Referencias

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298)
- [Documentación PyTorch](https://pytorch.org/docs/)

---

**Curso**: INFO1185 - Inteligencia Artificial  
**Profesor**: Prof. Dr. Ricardo Soto Catalán  
**Fecha**: Diciembre 2025
