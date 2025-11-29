"""
Script de entrenamiento rápido para probar tu dataset
"""

import torch
from config import get_config
from data_utils import get_data_transforms, create_data_loaders, show_sample_images
from models import create_model, print_model_summary
from training_utils import train_model, evaluate_model

def quick_train():
    """
    Entrenamiento rápido de un modelo para probar el dataset
    """
    
    print("🚀 ENTRENAMIENTO RÁPIDO - MODELO V1")
    print("="*50)
    
    # Configuración
    config = get_config('v1_simple')
    
    # Verificar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Dispositivo: {device}")
    
    # Cargar datos
    print("\n📁 Cargando dataset...")
    data_transforms = get_data_transforms(config)
    dataloaders, datasets, class_names = create_data_loaders(config, data_transforms)
    
    print(f"✅ Dataset cargado:")
    print(f"   📊 {len(class_names)} clases")
    print(f"   🎯 Train: {len(dataloaders['train'].dataset)} imágenes")
    print(f"   📋 Val: {len(dataloaders['val'].dataset)} imágenes")
    print(f"   🔍 Test: {len(dataloaders['test'].dataset)} imágenes")
    
    # Mostrar algunas clases
    print(f"\n🏷️ Clases incluidas:")
    for i, class_name in enumerate(class_names[:10]):  # Primeras 10
        print(f"   {i+1:2d}. {class_name}")
    if len(class_names) > 10:
        print(f"   ... y {len(class_names)-10} más")
    
    # Crear modelo
    print(f"\n🧠 Creando modelo V1 (ResNet18 + FC simple)...")
    model = create_model(
        model_name=config['model_name'],
        num_classes=len(class_names),
        version='simple',
        pretrained=config['pretrained']
    )
    
    print_model_summary(model)
    
    # Entrenar modelo
    print(f"\n🎯 Entrenando modelo (máx {config['epochs']} épocas)...")
    trained_model, history = train_model(
        model=model,
        dataloaders={'train': dataloaders['train'], 'val': dataloaders['val']},
        config=config,
        model_name="V1 - Prueba Rápida"
    )
    
    # Mostrar curvas
    history.plot_curves("Curvas de Entrenamiento - Prueba Rápida")
    
    # Evaluar en test
    print(f"\n🔍 Evaluando en conjunto de test...")
    test_results = evaluate_model(
        model=trained_model,
        dataloader=dataloaders['test'],
        class_names=class_names,
        phase_name="Test"
    )
    
    print(f"\n🏆 RESULTADO FINAL:")
    print(f"   ✅ Precisión en test: {test_results['accuracy']:.2%}")
    print(f"   📉 Pérdida en test: {test_results['loss']:.4f}")
    
    # Guardar modelo
    torch.save(trained_model.state_dict(), 'models/quick_test_model.pth')
    print(f"\n💾 Modelo guardado en: models/quick_test_model.pth")
    
    return trained_model, test_results

if __name__ == "__main__":
    # Ejecutar entrenamiento rápido
    model, results = quick_train()
    
    print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print("\n💡 PRÓXIMOS PASOS:")
    print("1. 📊 Ejecutar main.py para entrenar todos los modelos")
    print("2. 📋 Usar el notebook para análisis interactivo")
    print("3. 🎤 Preparar presentación con estos resultados")
