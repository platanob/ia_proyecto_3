import time
import copy
import torch
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device):
    """
    Entrena el modelo con early stopping y tracking de accuracy

    Returns:
        tuple: (best_model, train_losses, val_losses, train_accuracies, val_accuracies, best_epoch)
    """
    model = model.to(device)

    # Variables para tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    print(f"Iniciando entrenamiento...")
    print(f"Device: {device}")
    print(f"Epochs maximos: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Fase de entrenamiento
        model.train()
        running_loss = 0.0
        train_samples = 0
        train_correct = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Estadisticas
            running_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

            # Calcular accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                current_acc = 100.0 * train_correct / train_samples
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - Acc: {current_acc:.2f}%")

        # Métricas de entrenamiento
        epoch_train_loss = running_loss / train_samples
        epoch_train_accuracy = 100.0 * train_correct / train_samples
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        # Fase de validación
        model.eval()
        val_running_loss = 0.0
        val_samples = 0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        # Métricas de validación
        epoch_val_loss = val_running_loss / val_samples
        epoch_val_accuracy = 100.0 * val_correct / val_samples
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{num_epochs} completado en {epoch_time:.2f}s")
        print(
            f"Train - Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_accuracy:.2f}%")
        print(
            f"Val   - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_accuracy:.2f}%")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch + 1
            print(
                f"Nuevo mejor modelo Val Loss: {best_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%")
        else:
            patience_counter += 1
            print(f"Sin mejora. Paciencia: {patience_counter}/{patience}")

        print("-" * 60)

        if patience_counter >= patience:
            print(
                f"Early stopping activado! Sin mejora por {patience} epochs consecutivos.")
            break

    model.load_state_dict(best_model_state)
    total_time = time.time() - start_time

    print(f"\nEntrenamiento completado!")
    print(f"Tiempo total: {total_time/60:.2f} minutos")
    print(f"Mejor epoch: {best_epoch}")
    print(f"Mejor validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {val_accuracies[best_epoch-1]:.2f}%")
    print(f"Total de epochs: {len(train_losses)}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies, best_epoch


def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies, best_epoch=None):
    """
    Grafica las curvas de perdida y accuracy de entrenamiento y validacion
    """
    epochs = range(1, len(train_losses) + 1)

    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # SUBPLOT 1: LOSS
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss',
             linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss',
             linewidth=2, marker='s', markersize=4)

    if best_epoch:
        ax1.axvline(x=best_epoch, color='green', linestyle='--',
                    label=f'Best Epoch ({best_epoch})', alpha=0.7, linewidth=2)

    ax1.set_title('Curvas de Pérdida - EfficientNetV2-S V1',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # SUBPLOT 2: ACCURACY
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy',
             linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accuracies, 'orange',
             label='Validation Accuracy', linewidth=2, marker='s', markersize=4)

    if best_epoch:
        ax2.axvline(x=best_epoch, color='green', linestyle='--',
                    label=f'Best Epoch ({best_epoch})', alpha=0.7, linewidth=2)

    ax2.set_title('Curvas de Precisión - EfficientNetV2-S V1',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.show()

    # Estadísticas finales
    print("RESUMEN DE MÉTRICAS:")
    print("="*60)
    print(f"LOSS:")
    print(f"  • Final training loss:     {train_losses[-1]:.4f}")
    print(f"  • Final validation loss:   {val_losses[-1]:.4f}")
    print(
        f"  • Best validation loss:    {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")
    print(f"ACCURACY:")
    print(f"  • Final training accuracy:     {train_accuracies[-1]:.2f}%")
    print(f"  • Final validation accuracy:   {val_accuracies[-1]:.2f}%")
    print(
        f"  • Best validation accuracy:    {max(val_accuracies):.2f}% (epoch {val_accuracies.index(max(val_accuracies)) + 1})")

    if best_epoch:
        print(f"MEJOR MODELO (Epoch {best_epoch}):")
        print(f"  • Validation loss:     {val_losses[best_epoch-1]:.4f}")
        print(f"  • Validation accuracy: {val_accuracies[best_epoch-1]:.2f}%")
    print("="*60)
