# 🍎🥕 Transfer Learning para Clasificación de Frutas y Verduras

**Proyecto 3 - Inteligencia Artificial**  
**INFO1185** | **Prof. Dr. Ricardo Soto Catalán** | **Noviembre 2025**

## 📋 Descripción del Proyecto

Este proyecto implementa **Transfer Learning** usando PyTorch para clasificar imágenes de frutas y verduras. Se desarrollan y comparan dos variantes de clasificadores basados en modelos preentrenados, siguiendo los requerimientos específicos de la rúbrica del curso.

### 🎯 Objetivos

- ✅ Aplicar Transfer Learning con modelos preentrenados de `torchvision.models`
- ✅ Implementar dos variantes del clasificador (simple y embudo)
- ✅ Comparar el impacto de técnicas de regularización (Batch Normalization y Dropout)
- ✅ Evaluar modelos con métricas completas y visualizaciones
- ✅ Analizar resultados según criterios de la rúbrica

## 🏗️ Arquitectura del Proyecto

### Versión 1: Clasificador Simple
- **Estructura**: Backbone preentrenado + 1 capa FC
- **Características**: Sin Batch Normalization ni Dropout
- **Objetivo**: Baseline simple para comparación

### Versión 2: Clasificador Embudo  
- **Estructura**: Backbone preentrenado + arquitectura tipo embudo (ej: 512 → 256 → 128 → N)
- **Variantes**:
  - Sin regularización (sin BN ni Dropout)
  - Con regularización completa (BN + Dropout)
- **Objetivo**: Evaluar impacto de regularización

## 📁 Estructura del Proyecto

```
ia_proyecto_3/
├── 📓 Transfer_Learning_Frutas_Verduras.ipynb  # Notebook principal de Colab
├── 🐍 main.py                                 # Script principal ejecutable
├── ⚙️ config.py                              # Configuraciones del proyecto
├── 📊 data_utils.py                          # Utilidades para datos
├── 🧠 models.py                              # Definiciones de modelos
├── 🎯 training_utils.py                      # Utilidades de entrenamiento
├── 📖 README.md                              # Este archivo
├── 📋 presentation_template.md               # Plantilla para presentación
├── 📂 models/                                # Modelos entrenados guardados
├── 📈 results/                               # Resultados de experimentos
├── 📊 plots/                                 # Gráficos generados
└── 📁 fruits_vegetables_dataset/             # Dataset (descargar por separado)
```

## 🚀 Instalación y Configuración

### 1. Requisitos del Sistema
```bash
# Python 3.7+
# CUDA opcional (recomendado para GPU)
```

### 2. Instalación de Dependencias
```bash
# Instalar paquetes requeridos
pip install torch torchvision matplotlib seaborn scikit-learn pillow pandas numpy kaggle
```

### 3. Configuración del Dataset

#### Opción A: Usando Kaggle API (Recomendado)
```bash
# 1. Obtener credenciales de Kaggle
# Ve a tu cuenta de Kaggle → Account → API → Create New API Token
# Esto descarga kaggle.json

# 2. Configurar credenciales
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Descargar dataset
kaggle datasets download -d muhammadehsan02/fruits-and-vegetables-image-recognition-dataset

# 4. Extraer
unzip fruits-and-vegetables-image-recognition-dataset.zip -d fruits_vegetables_dataset
```

#### Opción B: Descarga Manual
1. Ve a [Kaggle Dataset](https://www.kaggle.com/datasets/muhammadehsan02/fruits-and-vegetables-image-recognition-dataset)
2. Descarga el dataset
3. Extrae en la carpeta `fruits_vegetables_dataset/`

### 4. Configuración del Proyecto
```python
# En config.py, actualiza la ruta del dataset:
PROJECT_CONFIG = {
    'data_dir': './fruits_vegetables_dataset',  # Tu ruta real
    'model_name': 'resnet18',                   # Modelo elegido
    # ... otras configuraciones
}
```

## 🖥️ Uso del Proyecto

### Opción 1: Ejecutar Todo (Recomendado)
```bash
python main.py
```

### Opción 2: Usar Google Colab
1. Sube `Transfer_Learning_Frutas_Verduras.ipynb` a Google Colab
2. Ejecuta todas las celdas secuencialmente
3. Configurar credenciales de Kaggle en Colab si es necesario

### Opción 3: Experimentos Individuales
```python
from main import run_experiment
from data_utils import get_data_transforms, create_data_loaders
from config import get_config

# Configurar datos
config = get_config('v1_simple')
data_transforms = get_data_transforms(config)
dataloaders, _, class_names = create_data_loaders(config, data_transforms)

# Ejecutar experimento específico
results = run_experiment('v1_simple', dataloaders, class_names)
```

## ⚙️ Configuración Detallada

### Modelos Disponibles
- `resnet18`, `resnet50`
- `vgg16`
- `densenet121`
- `mobilenet_v3_large`
- `efficientnet_v2_s`
- `googlenet`
- `inception_v3`
- `squeezenet1_1`

### Experimentos Definidos
```python
# v1_simple: Clasificador simple
# v2_no_reg: Embudo sin regularización
# v2_with_reg: Embudo con BN + Dropout
```

### Hiperparámetros Clave
```python
CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'patience': 10,        # Early stopping
    'dropout_rate': 0.3,   # Para versión 2
    'train_split': 0.7,    # 70% entrenamiento
    'val_split': 0.2,      # 20% validación  
    'test_split': 0.1      # 10% prueba
}
```

## 📊 Resultados y Evaluación

### Métricas Generadas
- ✅ **Precisión (Accuracy)** por clase y global
- ✅ **Precisión (Precision)** y **Sensibilidad (Recall)** por clase
- ✅ **Matriz de confusión** con visualización
- ✅ **Curvas de pérdida** durante entrenamiento
- ✅ **F1-Score** y reporte de clasificación completo

### Visualizaciones
- 📈 Curvas de entrenamiento (pérdida y precisión)
- 🎨 Matriz de confusión con heatmap
- 📊 Comparación entre modelos
- 🖼️ Muestras del dataset con predicciones

### Análisis Requerido (Según Rúbrica)
- 🔍 **Comparación entre V1 y V2**: Impacto de arquitectura embudo
- 🛡️ **Impacto de regularización**: BN + Dropout vs sin regularización
- ⚡ **Estabilidad del entrenamiento**: Análisis de curvas de pérdida
- 💻 **Limitaciones de Google Colab**: Memoria, tiempo de entrenamiento

## 🎯 Rúbrica del Proyecto

### Código (50 puntos)
- ✅ **Implementación del modelo** (10 pts): Modelos V1 y V2 correctos
- ✅ **Preparación de datos** (10 pts): DataLoaders con transformaciones
- ✅ **Entrenamientos requeridos** (10 pts): V1, V2-sin reg, V2-con reg
- ✅ **Métricas y gráficos** (10 pts): Curvas, matrices, métricas completas
- ✅ **Calidad del código** (10 pts): Limpio, comentado, ejecutable

### Presentación (60 puntos)
- 📋 **Descripción del problema** (5 pts)
- 📊 **Descripción del dataset** (5 pts)  
- 🧠 **Explicación del modelo** (10 pts)
- 📈 **Resultados y métricas** (10 pts)
- 🏁 **Conclusiones** (10 pts)
- 🎤 **Comunicación oral** (10 pts)
- ❓ **Respuesta a preguntas** (10 pts)

## 🐛 Solución de Problemas

### Error: Dataset no encontrado
```python
# Verifica la ruta en config.py
'data_dir': './fruits_vegetables_dataset'  # Ruta correcta

# O crea dataset de prueba
from data_utils import create_sample_dataset
create_sample_dataset()
```

### Error: Sin GPU en Colab
```python
# Runtime → Change runtime type → Hardware accelerator → GPU
# O usar CPU (más lento pero funcional)
```

### Error: Memoria insuficiente
```python
# Reduce batch_size en config.py
'batch_size': 16,  # En lugar de 32
```

## 📝 Para la Presentación

### Puntos Clave a Cubrir
1. **Problema**: Clasificación de frutas y verduras con Transfer Learning
2. **Dataset**: Tamaño, clases, distribución train/val/test
3. **Modelos**: Backbone elegido, V1 vs V2, arquitecturas
4. **Entrenamiento**: Hiperparámetros, early stopping, data augmentation
5. **Resultados**: Comparación cuantitativa y cualitativa
6. **Análisis**: Impacto de BN/Dropout, estabilidad, limitaciones
7. **Conclusiones**: Lecciones aprendidas, mejoras futuras

### Estructura Sugerida (8 minutos máx)
- 🎯 Introducción y objetivo (1 min)
- 📊 Dataset y preparación (1 min)  
- 🧠 Arquitecturas de modelos (2 min)
- 🎯 Estrategia de entrenamiento (1 min)
- 📈 Resultados y comparación (2 min)
- 🔍 Análisis y conclusiones (1 min)

## 👥 Información del Equipo

**Estudiantes**: [Agregar nombres aquí]  
**Modelo elegido**: [Agregar modelo elegido]  
**Problema de clasificación**: [Frutas/Verduras específicas]

## 📚 Referencias

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Fruits and Vegetables Dataset](https://www.kaggle.com/datasets/muhammadehsan02/fruits-and-vegetables-image-recognition-dataset)
- [Documentación PyTorch](https://pytorch.org/docs/)

---

**🎓 Curso**: INFO1185 - Inteligencia Artificial  
**👨‍🏫 Profesor**: Prof. Dr. Ricardo Soto Catalán  
**📅 Fecha límite**: 03 de diciembre de 2025, 13:00 hrs  
**🎤 Presentaciones**: 03 de diciembre de 2025, 13:50 hrs
