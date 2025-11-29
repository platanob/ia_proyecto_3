"""
Configuración del proyecto
Proyecto: Transfer Learning - Clasificación de Frutas y Verduras
"""

# Configuración principal del proyecto
PROJECT_CONFIG = {
    # ===============================
    # CONFIGURACIÓN DEL DATASET
    # ===============================
    'data_dir': './test',  # Carpeta con las imágenes organizadas por clase
    'batch_size': 16,  # Reducido para dataset pequeño
    'num_workers': 2,
    'train_split': 0.7,
    'val_split': 0.2,
    'test_split': 0.1,
    
    # ===============================
    # CONFIGURACIÓN DEL MODELO
    # ===============================
    # Opciones disponibles: 'resnet18', 'resnet50', 'vgg16', 'densenet121', 
    # 'mobilenet_v3_large', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'squeezenet1_1'
    'model_name': 'resnet18',
    'pretrained': True,
    'freeze_backbone': True,
    
    # ===============================
    # CONFIGURACIÓN DE ENTRENAMIENTO
    # ===============================
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 20,  # Reducido para dataset pequeño
    'patience': 5,  # Patience más bajo
    'min_delta': 0.001,
    
    # ===============================
    # REGULARIZACIÓN (VERSIÓN 2)
    # ===============================
    'dropout_rate': 0.3,
    'use_batch_norm': True,
    
    # ===============================
    # CONFIGURACIÓN DE IMAGEN
    # ===============================
    'image_size': 224,
    'normalize_mean': [0.485, 0.456, 0.406],  # Valores de ImageNet
    'normalize_std': [0.229, 0.224, 0.225],   # Valores de ImageNet
}

# Configuraciones específicas para diferentes experimentos
EXPERIMENT_CONFIGS = {
    # Versión 1: Clasificador Simple
    'v1_simple': {
        **PROJECT_CONFIG,
        'version': 'simple',
        'experiment_name': 'V1_Clasificador_Simple',
        'description': 'Versión 1: Una sola capa FC, sin BN ni Dropout'
    },
    
    # Versión 2: Sin regularización
    'v2_no_reg': {
        **PROJECT_CONFIG,
        'version': 'funnel',
        'use_batch_norm': False,
        'dropout_rate': 0.0,
        'experiment_name': 'V2_Sin_Regularizacion',
        'description': 'Versión 2: Embudo sin Batch Normalization ni Dropout'
    },
    
    # Versión 2: Con regularización completa
    'v2_with_reg': {
        **PROJECT_CONFIG,
        'version': 'funnel',
        'use_batch_norm': True,
        'dropout_rate': 0.3,
        'experiment_name': 'V2_Con_Regularizacion',
        'description': 'Versión 2: Embudo con Batch Normalization y Dropout'
    }
}

# Clases del dataset (basadas en tu carpeta test/)
FRUIT_VEGETABLE_CLASSES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 
    'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 
    'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
    'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 
    'mango', 'onion', 'orange', 'paprika', 'pear', 
    'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 
    'tomato', 'turnip', 'watermelon'
]

# Información del proyecto para la presentación
PROJECT_INFO = {
    'title': 'Transfer Learning para Clasificación de Frutas y Verduras',
    'course': 'INFO1185 - Inteligencia Artificial',
    'professor': 'Prof. Dr. Ricardo Soto Catalán',
    'date': 'Noviembre 2025',
    'objective': 'Aplicar Transfer Learning para clasificar imágenes de frutas y verduras usando PyTorch',
    'dataset_url': 'https://www.kaggle.com/datasets/muhammadehsan02/fruits-and-vegetables-image-recognition-dataset',
    'requirements': [
        'Dos variantes del clasificador (simple y embudo)',
        'Transfer Learning con modelos preentrenados',
        'Comparación con y sin técnicas de regularización',
        'Evaluación completa con métricas y visualizaciones'
    ]
}

# Funciones de utilidad para configuración
def get_config(experiment_name):
    """
    Obtener configuración para un experimento específico
    
    Args:
        experiment_name: Nombre del experimento ('v1_simple', 'v2_no_reg', 'v2_with_reg')
    
    Returns:
        dict: Configuración del experimento
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado. "
                        f"Opciones disponibles: {list(EXPERIMENT_CONFIGS.keys())}")
    
    return EXPERIMENT_CONFIGS[experiment_name]

def print_config(config):
    """
    Imprimir configuración de forma legible
    
    Args:
        config: Diccionario de configuración
    """
    print("⚙️ CONFIGURACIÓN DEL EXPERIMENTO")
    print("=" * 50)
    print(f"📋 Nombre: {config.get('experiment_name', 'Sin nombre')}")
    print(f"📝 Descripción: {config.get('description', 'Sin descripción')}")
    print()
    
    print("🗂️ DATASET:")
    print(f"  📁 Directorio: {config['data_dir']}")
    print(f"  🔄 Batch size: {config['batch_size']}")
    print(f"  📊 Train/Val/Test: {config['train_split']}/{config['val_split']}/{config['test_split']}")
    print()
    
    print("🧠 MODELO:")
    print(f"  🏗️ Arquitectura: {config['model_name']}")
    print(f"  📦 Preentrenado: {config['pretrained']}")
    print(f"  🔒 Backbone congelado: {config['freeze_backbone']}")
    print(f"  🏛️ Versión: {config.get('version', 'simple')}")
    print()
    
    print("🎯 ENTRENAMIENTO:")
    print(f"  📈 Learning rate: {config['learning_rate']}")
    print(f"  ⚖️ Weight decay: {config['weight_decay']}")
    print(f"  🔄 Épocas máx: {config['epochs']}")
    print(f"  ⏱️ Patience: {config['patience']}")
    print()
    
    if config.get('version') == 'funnel':
        print("🛡️ REGULARIZACIÓN:")
        print(f"  ✅ Batch Norm: {config['use_batch_norm']}")
        print(f"  💧 Dropout: {config['dropout_rate']}")
        print()
    
    print("=" * 50)

def validate_config(config):
    """
    Validar que la configuración sea correcta
    
    Args:
        config: Configuración a validar
    
    Returns:
        bool: True si es válida, False en caso contrario
    """
    required_keys = [
        'data_dir', 'batch_size', 'model_name', 'learning_rate', 
        'epochs', 'image_size'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Error: Clave requerida '{key}' no encontrada en configuración")
            return False
    
    # Validar splits
    total_split = config['train_split'] + config['val_split'] + config['test_split']
    if abs(total_split - 1.0) > 0.001:
        print(f"❌ Error: Los splits deben sumar 1.0, actual: {total_split}")
        return False
    
    # Validar modelo
    valid_models = [
        'resnet18', 'resnet50', 'vgg16', 'densenet121', 
        'mobilenet_v3_large', 'efficientnet_v2_s', 'googlenet', 
        'inception_v3', 'squeezenet1_1'
    ]
    if config['model_name'] not in valid_models:
        print(f"❌ Error: Modelo '{config['model_name']}' no válido. "
              f"Opciones: {valid_models}")
        return False
    
    print("✅ Configuración válida")
    return True

# Exportar configuraciones principales
__all__ = [
    'PROJECT_CONFIG',
    'EXPERIMENT_CONFIGS', 
    'FRUIT_VEGETABLE_CLASSES',
    'PROJECT_INFO',
    'get_config',
    'print_config',
    'validate_config'
]
