from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def evaluate_model(model, test_loader, class_names, device='cpu'):
    """
    Evalua el modelo en el conjunto de test

    Args:
        model: Modelo entrenado
        test_loader: DataLoader de test
        class_names (list): Nombres de las clases
        device (str): Dispositivo de computo

    Returns:
        tuple: (accuracy, y_true, y_pred)
    """
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    total_samples = 0
    correct_predictions = 0

    print("Evaluando modelo en conjunto de test...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_samples

    print(f"\nResultados de evaluacion:")
    print(f"Accuracy total: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Muestras correctas: {correct_predictions}/{total_samples}")

    # Matriz de confusion
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusion - EfficientNetV2-S V1',
              fontsize=16, fontweight='bold')
    plt.xlabel('Prediccion', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Reporte de clasificacion
    print("\nReporte de Clasificacion:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4, output_dict=False)
    print(report)

    return accuracy, y_true, y_pred
