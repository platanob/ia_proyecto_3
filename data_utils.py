"""
Utilidades para el procesamiento de datos
Proyecto: Transfer Learning - Clasificación de Frutas y Verduras
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from PIL import Image
from collections import Counter

class FruitVegetableDataset(Dataset):
    """Dataset personalizado para frutas y verduras"""
    
    def __init__(self, data_dir, transform=None, class_names=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.classes = class_names if class_names else self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self._load_samples()
    
    def _get_classes(self):
        """Obtiene las clases automáticamente de los directorios"""
        return sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
    
    def _load_samples(self):
        """Carga las muestras desde los directorios"""
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in self.classes:
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms(config):
    """
    Crear transformaciones para entrenamiento y validación/test
    
    Args:
        config: Diccionario con configuración
    
    Returns:
        dict: Transformaciones para train, val, test
    """
    
    # Transformaciones para entrenamiento (con data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(config['image_size'], scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['normalize_mean'], 
                           std=config['normalize_std'])
    ])
    
    # Transformaciones para validación y test (sin augmentation)
    val_test_transforms = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['normalize_mean'], 
                           std=config['normalize_std'])
    ])
    
    return {
        'train': train_transforms,
        'val': val_test_transforms,
        'test': val_test_transforms
    }

def create_data_loaders(config, data_transforms):
    """
    Crear DataLoaders para entrenamiento, validación y test
    
    Args:
        config: Diccionario con configuración
        data_transforms: Transformaciones para cada conjunto
    
    Returns:
        dict: DataLoaders para train, val, test
        dict: Datasets para train, val, test
        list: Nombres de las clases
    """
    
    # Cargar dataset completo
    full_dataset = datasets.ImageFolder(
        root=config['data_dir'],
        transform=data_transforms['train']
    )
    
    # Obtener clases
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Calcular tamaños de splits
    total_size = len(full_dataset)
    train_size = int(config['train_split'] * total_size)
    val_size = int(config['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    # Dividir dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Aplicar transformaciones específicas a cada conjunto
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    test_dataset.dataset.transform = data_transforms['test']
    
    # Crear DataLoaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    }
    
    datasets_dict = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    
    return dataloaders, datasets_dict, class_names

def visualize_dataset_info(dataloaders, class_names):
    """
    Visualizar información del dataset
    
    Args:
        dataloaders: DataLoaders del dataset
        class_names: Nombres de las clases
    """
    
    print("📊 INFORMACIÓN DEL DATASET")
    print("=" * 50)
    
    # Tamaños de conjuntos
    for phase in ['train', 'val', 'test']:
        size = len(dataloaders[phase].dataset)
        print(f"  {phase.capitalize():>6}: {size:,} imágenes")
    
    total = sum(len(loader.dataset) for loader in dataloaders.values())
    print(f"  {'Total':>6}: {total:,} imágenes")
    print()
    
    # Información de clases
    print(f"📋 CLASES ({len(class_names)} total):")
    for i, class_name in enumerate(class_names):
        print(f"  {i:2d}. {class_name}")
    print()

def show_sample_images(dataloader, class_names, num_samples=8):
    """
    Mostrar imágenes de muestra del dataset
    
    Args:
        dataloader: DataLoader del dataset
        class_names: Nombres de las clases
        num_samples: Número de imágenes a mostrar
    """
    
    # Obtener batch de imágenes
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Configurar subplot
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('🖼️ Muestras del Dataset', fontsize=16, fontweight='bold')
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i // 4, i % 4]
        
        # Desnormalizar imagen para visualización
        img = images[i]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Mostrar imagen
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f'{class_names[labels[i]]}', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_sample_dataset(base_dir='./sample_fruits_vegetables'):
    """
    Crear un dataset de muestra para pruebas
    NOTA: En la implementación real, usar el dataset de Kaggle
    
    Args:
        base_dir: Directorio base para crear el dataset
    """
    
    print("🚧 Creando dataset de muestra...")
    print("⚠️  IMPORTANTE: Esto es solo para demostración")
    print("   En la implementación real, usar el dataset de Kaggle")
    
    # Crear estructura de directorios
    classes = ['apple', 'banana', 'carrot', 'tomato', 'orange', 'broccoli']
    
    os.makedirs(base_dir, exist_ok=True)
    for class_name in classes:
        os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)
    
    print(f"📁 Estructura creada en: {base_dir}")
    print("   Directorios creados para las clases:", classes)
    print("\n💡 Para usar el dataset real:")
    print("   1. Descarga desde: https://www.kaggle.com/datasets/muhammadehsan02/fruits-and-vegetables-image-recognition-dataset")
    print("   2. Extrae en el directorio del proyecto")
    print("   3. Actualiza la ruta en CONFIG['data_dir']")
    
    return base_dir, classes
