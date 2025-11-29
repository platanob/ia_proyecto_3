"""
Utilidades para entrenamiento y evaluación
Proyecto: Transfer Learning - Clasificación de Frutas y Verduras
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class EarlyStopping:
    """
    Implementación de Early Stopping
    """
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())
    
    def restore_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

class TrainingHistory:
    """
    Clase para mantener el historial de entrenamiento
    """
    
    def __init__(self):
        self.history = defaultdict(list)
    
    def add(self, phase, epoch, loss, accuracy):
        self.history[f'{phase}_loss'].append(loss)
        self.history[f'{phase}_accuracy'].append(accuracy)
        self.history['epoch'].append(epoch)
    
    def plot_curves(self, title="Curvas de Entrenamiento"):
        """
        Plotear curvas de pérdida y precisión
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Curvas de pérdida
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Entrenamiento', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validación', linewidth=2)
        ax1.set_title('Pérdida durante el Entrenamiento')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Curvas de precisión
        ax2.plot(epochs, self.history['train_accuracy'], 'b-', label='Entrenamiento', linewidth=2)
        ax2.plot(epochs, self.history['val_accuracy'], 'r-', label='Validación', linewidth=2)
        ax2.set_title('Precisión durante el Entrenamiento')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Precisión (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def train_model(model, dataloaders, config, model_name="Modelo"):
    """
    Entrenar un modelo con early stopping
    
    Args:
        model: Modelo a entrenar
        dataloaders: DataLoaders para train y val
        config: Configuración de entrenamiento
        model_name: Nombre del modelo para logging
    
    Returns:
        tuple: (modelo_entrenado, historial)
    """
    
    print(f"🚀 INICIANDO ENTRENAMIENTO: {model_name}")
    print("=" * 60)
    
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizador y función de pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        min_delta=config['min_delta']
    )
    
    # Historial
    history = TrainingHistory()
    
    # Variables de seguimiento
    best_acc = 0.0
    start_time = time.time()
    
    # Bucle de entrenamiento
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        print(f'Época {epoch+1}/{config["epochs"]}')
        print('-' * 30)
        
        # Cada época tiene fase de entrenamiento y validación
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterar sobre los datos
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Limpiar gradientes
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass solo en entrenamiento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Estadísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calcular métricas de la época
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Guardar en historial
            history.add(phase, epoch, epoch_loss, epoch_acc.item() * 100)
            
            # Guardar mejor modelo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                
        # Early stopping
        early_stopping(history.history['val_loss'][-1], model)
        
        epoch_time = time.time() - epoch_start_time
        print(f'Tiempo de época: {epoch_time:.2f}s')
        print()
        
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping en época {epoch+1}")
            break
    
    # Restaurar mejores pesos
    early_stopping.restore_weights(model)
    
    total_time = time.time() - start_time
    print(f'✅ Entrenamiento completado en {total_time//60:.0f}m {total_time%60:.0f}s')
    print(f'Mejor precisión de validación: {best_acc:.4f}')
    
    return model, history

def evaluate_model(model, dataloader, class_names, phase_name="Test"):
    """
    Evaluar un modelo en el conjunto de test
    
    Args:
        model: Modelo entrenado
        dataloader: DataLoader del conjunto de test
        class_names: Nombres de las clases
        phase_name: Nombre de la fase (para logging)
    
    Returns:
        dict: Métricas de evaluación
    """
    
    print(f"🔍 EVALUACIÓN EN {phase_name.upper()}")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas globales
    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f"Pérdida: {total_loss:.4f}")
    print(f"Precisión: {total_acc:.4f}")
    print()
    
    # Reporte de clasificación detallado
    print("📊 REPORTE DE CLASIFICACIÓN")
    print("-" * 40)
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, f"Matriz de Confusión - {phase_name}")
    
    return {
        'loss': total_loss,
        'accuracy': total_acc.item(),
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_confusion_matrix(cm, class_names, title="Matriz de Confusión"):
    """
    Plotear matriz de confusión
    
    Args:
        cm: Matriz de confusión
        class_names: Nombres de las clases
        title: Título del gráfico
    """
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Número de Predicciones'}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Verdadero', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def compare_models(results_dict, metric='accuracy'):
    """
    Comparar resultados de múltiples modelos
    
    Args:
        results_dict: Diccionario con resultados {nombre_modelo: resultados}
        metric: Métrica a comparar ('accuracy', 'loss')
    """
    
    print(f"📊 COMPARACIÓN DE MODELOS - {metric.upper()}")
    print("=" * 50)
    
    # Extraer datos para comparación
    names = []
    values = []
    
    for name, results in results_dict.items():
        names.append(name)
        values.append(results[metric])
    
    # Crear gráfico
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, values, alpha=0.8, edgecolor='black')
    
    # Personalizar gráfico
    plt.title(f'Comparación de Modelos - {metric.title()}', fontsize=16, fontweight='bold')
    plt.ylabel(metric.title())
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Resaltar el mejor
    best_idx = values.index(max(values) if metric == 'accuracy' else min(values))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    
    plt.tight_layout()
    plt.show()
    
    # Tabla de resultados
    print("\n📋 TABLA DE RESULTADOS")
    print("-" * 30)
    sorted_results = sorted(results_dict.items(), 
                          key=lambda x: x[1][metric], 
                          reverse=(metric == 'accuracy'))
    
    for i, (name, results) in enumerate(sorted_results):
        symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{symbol} {name}: {results[metric]:.4f}")

def save_model(model, path, additional_info=None):
    """
    Guardar modelo entrenado
    
    Args:
        model: Modelo a guardar
        path: Ruta donde guardar
        additional_info: Información adicional para guardar
    """
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'model_type': model.__class__.__name__,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, path)
    print(f"💾 Modelo guardado en: {path}")

def load_model(model_class, path, **kwargs):
    """
    Cargar modelo entrenado
    
    Args:
        model_class: Clase del modelo
        path: Ruta del archivo del modelo
        **kwargs: Argumentos para crear el modelo
    
    Returns:
        Modelo cargado
    """
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # Crear modelo
    model = model_class(**kwargs)
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📂 Modelo cargado desde: {path}")
    return model
