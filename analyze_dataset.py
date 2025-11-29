"""
Script para analizar el dataset en la carpeta test/
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset(data_dir='./test'):
    """
    Analizar el dataset y mostrar estadísticas
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: No se encuentra la carpeta {data_dir}")
        return None
    
    print(f"📁 ANÁLISIS DEL DATASET: {data_dir}")
    print("="*50)
    
    # Obtener clases
    classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    
    print(f"🏷️ CLASES ENCONTRADAS ({len(classes)} total):")
    print("-"*30)
    
    class_counts = {}
    total_images = 0
    
    for i, class_name in enumerate(classes):
        class_path = data_path / class_name
        
        # Contar imágenes en esta clase
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = [f for f in class_path.iterdir() 
                 if f.suffix.lower() in image_extensions]
        
        count = len(images)
        class_counts[class_name] = count
        total_images += count
        
        print(f"  {i+1:2d}. {class_name:15} : {count:4d} imágenes")
    
    print("-"*30)
    print(f"📊 TOTAL: {total_images:,} imágenes en {len(classes)} clases")
    
    # Estadísticas
    counts = list(class_counts.values())
    if counts:
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   Promedio por clase: {sum(counts)/len(counts):.1f}")
        print(f"   Clase con más imágenes: {max(class_counts, key=class_counts.get)} ({max(counts)})")
        print(f"   Clase con menos imágenes: {min(class_counts, key=class_counts.get)} ({min(counts)})")
    
    # Verificar distribución para splits
    print(f"\n🔄 SPLITS SUGERIDOS:")
    print(f"   70% Entrenamiento: {int(total_images * 0.7):,} imágenes")
    print(f"   20% Validación: {int(total_images * 0.2):,} imágenes") 
    print(f"   10% Prueba: {int(total_images * 0.1):,} imágenes")
    
    # Crear gráfico de distribución
    if len(classes) <= 20:  # Solo si no hay demasiadas clases
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(classes)), counts)
        plt.title('Distribución de Imágenes por Clase')
        plt.xlabel('Clases')
        plt.ylabel('Número de Imágenes')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    return {
        'classes': classes,
        'class_counts': class_counts,
        'total_images': total_images
    }

def check_sample_images(data_dir='./test', num_samples=3):
    """
    Verificar algunas imágenes de muestra
    """
    data_path = Path(data_dir)
    classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    print(f"\n🖼️ VERIFICANDO IMÁGENES DE MUESTRA")
    print("-"*40)
    
    sample_classes = classes[:min(4, len(classes))]  # Primeras 4 clases
    
    for class_name in sample_classes:
        class_path = data_path / class_name
        image_files = [f for f in class_path.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        print(f"📂 {class_name}:")
        for i, img_file in enumerate(image_files[:num_samples]):
            try:
                from PIL import Image
                img = Image.open(img_file)
                print(f"   ✅ {img_file.name} - {img.size} - {img.mode}")
            except Exception as e:
                print(f"   ❌ {img_file.name} - Error: {e}")
        print()

if __name__ == "__main__":
    # Analizar el dataset
    results = analyze_dataset('./test')
    
    # Verificar muestras
    check_sample_images('./test')
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("1. ✅ Dataset detectado y analizado")
    print("2. 🎯 Ejecutar main.py para entrenar los modelos") 
    print("3. 📊 O usar el notebook para entrenamiento interactivo")
