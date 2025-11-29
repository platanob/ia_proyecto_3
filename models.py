"""
Definición de modelos para Transfer Learning
Proyecto: Transfer Learning - Clasificación de Frutas y Verduras
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TransferLearningModel(nn.Module):
    """
    Clase base para modelos de Transfer Learning
    """
    
    def __init__(self, model_name, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Cargar modelo preentrenado
        self.backbone = self._get_backbone(model_name, pretrained)
        self.feature_size = self._get_feature_size()
        
        # Congelar backbone
        self._freeze_backbone()
    
    def _get_backbone(self, model_name, pretrained):
        """Obtener el backbone según el nombre del modelo"""
        
        model_dict = {
            'resnet18': models.resnet18(pretrained=pretrained),
            'resnet50': models.resnet50(pretrained=pretrained),
            'vgg16': models.vgg16(pretrained=pretrained),
            'densenet121': models.densenet121(pretrained=pretrained),
            'mobilenet_v3_large': models.mobilenet_v3_large(pretrained=pretrained),
            'efficientnet_v2_s': models.efficientnet_v2_s(pretrained=pretrained),
            'googlenet': models.googlenet(pretrained=pretrained),
            'inception_v3': models.inception_v3(pretrained=pretrained),
            'squeezenet1_1': models.squeezenet1_1(pretrained=pretrained),
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Modelo {model_name} no soportado. Opciones: {list(model_dict.keys())}")
        
        return model_dict[model_name]
    
    def _get_feature_size(self):
        """Obtener el tamaño de las características del backbone"""
        
        if 'resnet' in self.model_name:
            return self.backbone.fc.in_features
        elif 'vgg' in self.model_name:
            return self.backbone.classifier[0].in_features
        elif 'densenet' in self.model_name:
            return self.backbone.classifier.in_features
        elif 'mobilenet' in self.model_name:
            return self.backbone.classifier[0].in_features
        elif 'efficientnet' in self.model_name:
            return self.backbone.classifier[1].in_features
        elif 'googlenet' in self.model_name or 'inception' in self.model_name:
            return self.backbone.fc.in_features
        elif 'squeezenet' in self.model_name:
            return 512  # SqueezeNet tiene una estructura diferente
        else:
            raise ValueError(f"No se puede determinar feature_size para {self.model_name}")
    
    def _freeze_backbone(self):
        """Congelar los pesos del backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _remove_final_layer(self):
        """Remover la capa final del backbone"""
        if 'resnet' in self.model_name:
            self.backbone.fc = nn.Identity()
        elif 'vgg' in self.model_name:
            self.backbone.classifier = self.backbone.classifier[:-1]
        elif 'densenet' in self.model_name:
            self.backbone.classifier = nn.Identity()
        elif 'mobilenet' in self.model_name:
            self.backbone.classifier = nn.Identity()
        elif 'efficientnet' in self.model_name:
            self.backbone.classifier = nn.Identity()
        elif 'googlenet' in self.model_name or 'inception' in self.model_name:
            self.backbone.fc = nn.Identity()
        elif 'squeezenet' in self.model_name:
            self.backbone.classifier = nn.Identity()

class SimpleClassifier(TransferLearningModel):
    """
    VERSIÓN 1: Clasificador simple con una sola capa FC
    Según la rúbrica: solo una capa FC, sin BN ni Dropout
    """
    
    def __init__(self, model_name, num_classes, pretrained=True):
        super(SimpleClassifier, self).__init__(model_name, num_classes, pretrained)
        
        # Remover capa final del backbone
        self._remove_final_layer()
        
        # Clasificador simple: solo una capa FC
        self.classifier = nn.Linear(self.feature_size, num_classes)
        
        print(f"📋 MODELO V1 - Clasificador Simple")
        print(f"   Backbone: {model_name}")
        print(f"   Características: {self.feature_size}")
        print(f"   Clases: {num_classes}")
        print(f"   Arquitectura: {self.feature_size} → {num_classes}")
    
    def forward(self, x):
        # Extraer características del backbone
        features = self.backbone(x)
        
        # Para SqueezeNet, manejar la salida especial
        if 'squeezenet' in self.model_name:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # Clasificación
        output = self.classifier(features)
        
        return output

class FunnelClassifier(TransferLearningModel):
    """
    VERSIÓN 2: Clasificador extendido tipo embudo
    Según la rúbrica: arquitectura embudo con BN, Dropout y activaciones no lineales
    """
    
    def __init__(self, model_name, num_classes, pretrained=True, 
                 use_batch_norm=True, dropout_rate=0.3):
        super(FunnelClassifier, self).__init__(model_name, num_classes, pretrained)
        
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Remover capa final del backbone
        self._remove_final_layer()
        
        # Arquitectura tipo embudo (ejemplo: 512 → 256 → 128 → num_classes)
        hidden_sizes = self._calculate_funnel_sizes()
        
        # Construir clasificador
        layers = []
        input_size = self.feature_size
        
        for hidden_size in hidden_sizes:
            # Capa lineal
            layers.append(nn.Linear(input_size, hidden_size))
            
            # Batch Normalization (si se especifica)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activación ReLU
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout (si se especifica)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            input_size = hidden_size
        
        # Capa final (sin activación)
        layers.append(nn.Linear(input_size, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Mostrar información
        arch_str = " → ".join([str(self.feature_size)] + [str(h) for h in hidden_sizes] + [str(num_classes)])
        print(f"📋 MODELO V2 - Clasificador Embudo")
        print(f"   Backbone: {model_name}")
        print(f"   Características: {self.feature_size}")
        print(f"   Clases: {num_classes}")
        print(f"   Arquitectura: {arch_str}")
        print(f"   Batch Normalization: {use_batch_norm}")
        print(f"   Dropout: {dropout_rate}")
    
    def _calculate_funnel_sizes(self):
        """Calcular tamaños para arquitectura tipo embudo"""
        
        # Estrategia: reducir progresivamente hasta llegar a un tamaño razonable
        sizes = []
        current_size = self.feature_size
        
        # Primera capa oculta: reducir a la mitad
        if current_size > 512:
            sizes.append(512)
            current_size = 512
        
        # Segunda capa: reducir a la mitad nuevamente
        if current_size > 256:
            sizes.append(256)
            current_size = 256
        
        # Tercera capa: reducir a 128 si es necesario
        if current_size > 128 and self.num_classes < 128:
            sizes.append(128)
        
        # Asegurar al menos 2 capas ocultas como requiere la rúbrica
        if len(sizes) < 2:
            sizes = [max(256, current_size // 2), max(128, current_size // 4)]
        
        return sizes
    
    def forward(self, x):
        # Extraer características del backbone
        features = self.backbone(x)
        
        # Para SqueezeNet, manejar la salida especial
        if 'squeezenet' in self.model_name:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # Clasificación con arquitectura embudo
        output = self.classifier(features)
        
        return output

def create_model(model_name, num_classes, version='simple', **kwargs):
    """
    Función factory para crear modelos
    
    Args:
        model_name: Nombre del modelo backbone
        num_classes: Número de clases
        version: 'simple' para V1, 'funnel' para V2
        **kwargs: Argumentos adicionales para el modelo
    
    Returns:
        Modelo inicializado
    """
    
    if version == 'simple':
        return SimpleClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=kwargs.get('pretrained', True)
        )
    elif version == 'funnel':
        return FunnelClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=kwargs.get('pretrained', True),
            use_batch_norm=kwargs.get('use_batch_norm', True),
            dropout_rate=kwargs.get('dropout_rate', 0.3)
        )
    else:
        raise ValueError("version debe ser 'simple' o 'funnel'")

def print_model_summary(model, input_size=(3, 224, 224)):
    """
    Imprimir resumen del modelo
    
    Args:
        model: Modelo a analizar
        input_size: Tamaño de entrada (C, H, W)
    """
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("🔍 RESUMEN DEL MODELO")
    print("=" * 50)
    print(f"Arquitectura: {model.__class__.__name__}")
    print(f"Backbone: {model.model_name}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Parámetros congelados: {frozen_params:,}")
    print(f"Porcentaje entrenable: {100 * trainable_params / total_params:.1f}%")
    
    # Probar forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        try:
            output = model(dummy_input)
            print(f"Entrada: {dummy_input.shape}")
            print(f"Salida: {output.shape}")
        except Exception as e:
            print(f"Error en forward pass: {e}")
    
    print("=" * 50)
