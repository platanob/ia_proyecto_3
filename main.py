"""
Script principal para ejecutar todos los experimentos
Proyecto: Transfer Learning - Clasificación de Frutas y Verduras
"""

import os
import torch
from pathlib import Path

# Importar módulos del proyecto
from config import EXPERIMENT_CONFIGS, get_config, print_config, validate_config
from data_utils import get_data_transforms, create_data_loaders, visualize_dataset_info, show_sample_images
from models import create_model, print_model_summary
from training_utils import train_model, evaluate_model, compare_models, save_model

def setup_directories():
    """Crear directorios necesarios para el proyecto"""
    
    directories = [
        'models',
        'results', 
        'plots',
        'logs'
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Directorio creado/verificado: {dir_name}")

def run_experiment(experiment_name, dataloaders, class_names, save_results=True):
    """
    Ejecutar un experimento completo
    
    Args:
        experiment_name: Nombre del experimento
        dataloaders: DataLoaders del dataset
        class_names: Nombres de las clases
        save_results: Si guardar resultados
    
    Returns:
        dict: Resultados del experimento
    """
    
    print(f"\n🚀 INICIANDO EXPERIMENTO: {experiment_name.upper()}")
    print("="*60)
    
    # Obtener configuración
    config = get_config(experiment_name)
    print_config(config)
    
    if not validate_config(config):
        print("❌ Configuración inválida, saltando experimento")
        return None
    
    # Crear modelo
    print("\n🏗️ CREANDO MODELO...")
    model = create_model(
        model_name=config['model_name'],
        num_classes=len(class_names),
        version=config['version'],
        pretrained=config['pretrained'],
        use_batch_norm=config.get('use_batch_norm', True),
        dropout_rate=config.get('dropout_rate', 0.0)
    )
    
    # Mostrar resumen del modelo
    print_model_summary(model)
    
    # Entrenar modelo
    print("\n🎯 ENTRENANDO MODELO...")
    trained_model, history = train_model(
        model=model,
        dataloaders={'train': dataloaders['train'], 'val': dataloaders['val']},
        config=config,
        model_name=config['experiment_name']
    )
    
    # Mostrar curvas de entrenamiento
    history.plot_curves(title=f"Curvas de Entrenamiento - {config['experiment_name']}")
    
    # Evaluar en conjunto de test
    print("\n🔍 EVALUANDO MODELO...")
    test_results = evaluate_model(
        model=trained_model,
        dataloader=dataloaders['test'],
        class_names=class_names,
        phase_name="Test"
    )
    
    # Preparar resultados
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'model': trained_model,
        'history': history,
        'test_results': test_results,
        'accuracy': test_results['accuracy'],
        'loss': test_results['loss']
    }
    
    # Guardar modelo si se especifica
    if save_results:
        model_path = f"models/{experiment_name}_model.pth"
        save_model(
            model=trained_model,
            path=model_path,
            additional_info={
                'config': config,
                'test_accuracy': test_results['accuracy'],
                'class_names': class_names
            }
        )
        results['model_path'] = model_path
    
    print(f"✅ EXPERIMENTO {experiment_name.upper()} COMPLETADO")
    print(f"   Precisión en test: {test_results['accuracy']:.4f}")
    print("="*60)
    
    return results

def run_all_experiments():
    """
    Ejecutar todos los experimentos requeridos por la rúbrica
    """
    
    print("🍎🥕 PROYECTO: TRANSFER LEARNING - CLASIFICACIÓN DE FRUTAS Y VERDURAS")
    print("="*80)
    print("📚 Curso: INFO1185 - Inteligencia Artificial")
    print("👨‍🏫 Profesor: Prof. Dr. Ricardo Soto Catalán")
    print("📅 Fecha: Noviembre 2025")
    print("="*80)
    
    # Configurar directorios
    setup_directories()
    
    # Verificar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Dispositivo: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Obtener configuración base para datos
    base_config = get_config('v1_simple')
    
    # Crear transformaciones y cargar datos
    print("📁 PREPARANDO DATASET...")
    data_transforms = get_data_transforms(base_config)
    
    try:
        dataloaders, datasets, class_names = create_data_loaders(base_config, data_transforms)
        
        # Mostrar información del dataset
        visualize_dataset_info(dataloaders, class_names)
        
        # Mostrar muestras del dataset
        show_sample_images(dataloaders['train'], class_names)
        
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        print("💡 Asegúrate de haber descargado el dataset de Kaggle y actualizado la ruta en config.py")
        return
    
    # Ejecutar todos los experimentos
    results = {}
    
    # Lista de experimentos según la rúbrica
    experiments = [
        'v1_simple',        # Versión 1: Clasificador simple
        'v2_no_reg',        # Versión 2: Sin regularización  
        'v2_with_reg'       # Versión 2: Con regularización
    ]
    
    for exp_name in experiments:
        try:
            result = run_experiment(
                experiment_name=exp_name,
                dataloaders=dataloaders,
                class_names=class_names,
                save_results=True
            )
            if result:
                results[result['experiment_name']] = result
        except Exception as e:
            print(f"❌ Error en experimento {exp_name}: {e}")
            continue
    
    # Comparar resultados
    if len(results) > 1:
        print("\n📊 COMPARACIÓN FINAL DE RESULTADOS")
        print("="*50)
        
        # Preparar datos para comparación
        comparison_data = {}
        for name, result in results.items():
            comparison_data[result['config']['experiment_name']] = {
                'accuracy': result['accuracy'],
                'loss': result['loss']
            }
        
        # Mostrar comparaciones
        compare_models(comparison_data, 'accuracy')
        compare_models(comparison_data, 'loss')
        
        # Resumen final
        print("\n🏆 RESUMEN FINAL")
        print("-" * 30)
        
        best_acc = max(results.values(), key=lambda x: x['accuracy'])
        print(f"🥇 Mejor precisión: {best_acc['config']['experiment_name']} ({best_acc['accuracy']:.4f})")
        
        # Análisis según la rúbrica
        print("\n📝 ANÁLISIS PARA LA RÚBRICA:")
        print("-" * 40)
        
        if 'V1_Clasificador_Simple' in comparison_data and 'V2_Con_Regularizacion' in comparison_data:
            v1_acc = comparison_data['V1_Clasificador_Simple']['accuracy']
            v2_acc = comparison_data['V2_Con_Regularizacion']['accuracy']
            
            print(f"• Versión 1 (Simple): {v1_acc:.4f}")
            print(f"• Versión 2 (Embudo): {v2_acc:.4f}")
            print(f"• Mejora con arquitectura embudo: {((v2_acc - v1_acc) / v1_acc * 100):+.2f}%")
        
        if 'V2_Sin_Regularizacion' in comparison_data and 'V2_Con_Regularizacion' in comparison_data:
            no_reg_acc = comparison_data['V2_Sin_Regularizacion']['accuracy']
            with_reg_acc = comparison_data['V2_Con_Regularizacion']['accuracy']
            
            print(f"• V2 sin regularización: {no_reg_acc:.4f}")
            print(f"• V2 con regularización: {with_reg_acc:.4f}")
            print(f"• Impacto de BN + Dropout: {((with_reg_acc - no_reg_acc) / no_reg_acc * 100):+.2f}%")
    
    print("\n✅ TODOS LOS EXPERIMENTOS COMPLETADOS")
    print("📁 Revisa las carpetas 'models', 'results' y 'plots' para los archivos generados")
    print("📋 Usa estos resultados para crear tu presentación según la rúbrica")

if __name__ == "__main__":
    run_all_experiments()
