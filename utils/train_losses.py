import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_with_loss_tracking(model, train_loader, val_loader, num_epochs=10,
                                   learning_rate=0.001, weight_decay=1e-4):
    """
    Función general para entrenar un modelo y calcular curvas de pérdida

    Args:
        model: Modelo PyTorch a entrenar
        train_loader: DataLoader para datos de entrenamiento
        val_loader: DataLoader para datos de validación
        num_epochs: Número de épocas
        learning_rate: Tasa de aprendizaje
        weight_decay: Regularización L2

    Returns:
        train_losses: Lista de pérdidas de entrenamiento por época
        val_losses: Lista de pérdidas de validación por época
        train_accuracies: Lista de precisiones de entrenamiento por época
        val_accuracies: Lista de precisiones de validación por época
    """

    # Configurar optimizador y función de pérdida
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Listas para almacenar métricas
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Iniciando entrenamiento...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # ============ ENTRENAMIENTO ============
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Mostrar progreso cada 10 batches
            if batch_idx % 10 == 0:
                print(f'Época {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # Calcular métricas promedio de entrenamiento
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train

        # ============ VALIDACIÓN ============
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calcular métricas promedio de validación
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * correct_val / total_val

        # Guardar métricas
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        # Mostrar resumen de la época
        print(f'Época {epoch+1}/{num_epochs}:')
        print(
            f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(
            f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print("-" * 60)

    return train_losses, val_losses, train_accuracies, val_accuracies


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies,
                         model_name="Modelo", save_path=None):
    """
    Función para visualizar las curvas de pérdida y precisión

    Args:
        train_losses: Lista de pérdidas de entrenamiento
        val_losses: Lista de pérdidas de validación
        train_accuracies: Lista de precisiones de entrenamiento
        val_accuracies: Lista de precisiones de validación
        model_name: Nombre del modelo para el título
        save_path: Ruta para guardar la imagen (opcional)
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de pérdidas
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'bo-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss', linewidth=2)
    ax1.set_title(f'Curvas de Pérdida - {model_name}')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico de precisión
    ax2.plot(epochs, train_accuracies, 'bo-',
             label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'ro-',
             label='Validation Accuracy', linewidth=2)
    ax2.set_title(f'Curvas de Precisión - {model_name}')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráficos guardados en: {save_path}")

    plt.show()


def print_prediction_summary(predictions, image_path, top_k=5):
    """ 
    Imprime un resumen detallado de las predicciones 
    """
    print("\n" + "="*60)
    print(f"PREDICCIÓN PARA: {os.path.basename(image_path)}")
    print("="*60)
    if top_k:
        display_predictions = predictions[:top_k]
    else:
        display_predictions = predictions
    for i, (class_name, probability) in enumerate(display_predictions):
        status = "⭐ PREDICCIÓN PRINCIPAL" if i == 0 else "  "
        print(f"{i+1:2d}. {class_name:<20} {probability:6.2f}% {status}")
    print("="*60)


def print_training_summary(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Función para imprimir un resumen del entrenamiento
    """
    print("\n" + "="*50)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*50)
    print(f"Pérdida final de entrenamiento: {train_losses[-1]:.4f}")
    print(f"Pérdida final de validación: {val_losses[-1]:.4f}")
    print(f"Precisión final de entrenamiento: {train_accuracies[-1]:.2f}%")
    print(f"Precisión final de validación: {val_accuracies[-1]:.2f}%")

    best_val_epoch = val_accuracies.index(max(val_accuracies)) + 1
    print(
        f"Mejor precisión de validación: {max(val_accuracies):.2f}% (Época {best_val_epoch})")

    min_val_loss_epoch = val_losses.index(min(val_losses)) + 1
    print(
        f"Menor pérdida de validación: {min(val_losses):.4f} (Época {min_val_loss_epoch})")
    print("="*50)
