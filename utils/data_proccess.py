from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def create_transforms(img_size=224):
    """
    Crea las transformaciones para entrenamiento y validacion/test

    Args:
        img_size (int): Tamaño de imagen (default: 224)

    Returns:
        tuple: (train_transform, val_test_transform)
    """
    # Normalizacion de ImageNet
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # Transformaciones para entrenamiento (con data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # Transformaciones para validacion y test (sin augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_test_transform


def load_datasets(data_dir, train_transform, val_test_transform):
    """
    Carga los datasets desde la estructura de directorios

    Args:
        data_dir (str): Directorio raiz de los datos
        train_transform: Transformaciones para entrenamiento
        val_test_transform: Transformaciones para validacion/test

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/validation",
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=val_test_transform
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Crea los DataLoaders para entrenamiento, validacion y test

    Args:
        train_dataset: Dataset de entrenamiento
        val_dataset: Dataset de validacion
        test_dataset: Dataset de test
        batch_size (int): Tamaño del batch

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
