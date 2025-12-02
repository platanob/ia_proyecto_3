import torch
import torch.nn.functional as F
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


def predict_single_image(model, image_path, class_names, transform, device):
    """
    Predice la clase de una imagen individual

    Args:
        model: Modelo entrenado
        image_path: Ruta a la imagen
        class_names: Lista de nombres de las clases
        transform: Transformaciones a aplicar
        device: Dispositivo de computo

    Returns:
        tuple: (predictions, predicted_class, confidence, original_image)
    """
    try:
        # Cargar y procesar la imagen
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()

        # Aplicar transformaciones
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Realizar prediccion
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

            # Obtener todas las probabilidades
            probs = probabilities.cpu().numpy()[0]

            # Crear lista de predicciones ordenadas
            predictions = [(class_names[i], prob * 100)
                           for i, prob in enumerate(probs)]
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Clase con mayor probabilidad
            predicted_class = predictions[0][0]
            confidence = predictions[0][1]

        return predictions, predicted_class, confidence, original_image

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None, None, None, None


def visualize_prediction_result(image, predictions, predicted_class, image_path, top_k=5):
    """
    Visualiza la imagen y las predicciones
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Mostrar imagen original
    ax1.imshow(image)
    ax1.set_title(f'Imagen: {os.path.basename(image_path)}\nPrediccion: {predicted_class}',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Mostrar grafico de barras con probabilidades
    top_predictions = predictions[:top_k]
    classes = [pred[0] for pred in top_predictions]
    probs = [pred[1] for pred in top_predictions]

    # Colores para las barras
    colors = ['#2E86AB' if i == 0 else '#A23B72' for i in range(len(classes))]

    bars = ax2.barh(range(len(classes)), probs, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes, fontsize=12)
    ax2.set_xlabel('Probabilidad (%)', fontsize=12)
    ax2.set_title(f'Top {len(top_predictions)} Predicciones',
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Agregar valores en las barras
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 1, bar.get_y() + bar.get_height()/2,
                 f'{prob:.1f}%', va='center', ha='left',
                 fontweight='bold' if i == 0 else 'normal')

    plt.tight_layout()
    plt.show()


def print_detailed_prediction(predictions, image_path, top_k=5):
    """
    Imprime resumen detallado de las predicciones
    """
    print("\n" + "="*70)
    print(f"PREDICCION PARA: {os.path.basename(image_path)}")
    print("="*70)

    display_predictions = predictions[:top_k]
    for i, (class_name, probability) in enumerate(display_predictions):
        status = "⭐ PREDICCION PRINCIPAL" if i == 0 else ""
        print(f"{i+1:2d}. {class_name:<15} {probability:6.2f}% {status}")

    print("="*70)


def predict_random_from_test(model, test_dataset_path, class_names, transform, device, num_samples=3):
    """
    Selecciona imagenes aleatorias del conjunto de test y realiza predicciones
    """
    # Obtener todas las clases disponibles
    available_classes = [d for d in os.listdir(test_dataset_path)
                         if os.path.isdir(os.path.join(test_dataset_path, d))]

    print(f"Clases disponibles en test: {available_classes}")

    # Seleccionar imagenes aleatorias
    selected_images = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for class_folder in available_classes:
        class_path = os.path.join(test_dataset_path, class_folder)
        images = [f for f in os.listdir(class_path)
                  if os.path.splitext(f.lower())[1] in valid_extensions]

        if images:
            # Seleccionar una imagen aleatoria de esta clase
            selected_image = random.choice(images)
            selected_images.append(os.path.join(class_path, selected_image))

    # Limitar al numero de muestras solicitado
    selected_images = selected_images[:num_samples]

    print(
        f"\nAnalizando {len(selected_images)} imagenes aleatorias del conjunto de test...")

    # Procesar cada imagen
    for i, image_path in enumerate(selected_images):
        print(f"\n{'='*25} IMAGEN {i+1}/{len(selected_images)} {'='*25}")

        predictions, predicted_class, confidence, original_image = predict_single_image(
            model, image_path, class_names, transform, device
        )

        if predictions:
            # Obtener la clase real del path
            real_class = os.path.basename(os.path.dirname(image_path))

            print(f"Clase real: {real_class}")
            print(
                f"Prediccion: {predicted_class} (Confianza: {confidence:.2f}%)")

            # Verificar si la prediccion es correcta
            is_correct = "✅ CORRECTA" if predicted_class == real_class else "❌ INCORRECTA"
            print(f"Resultado: {is_correct}")

            visualize_prediction_result(
                original_image, predictions, predicted_class, image_path, top_k=len(class_names))
            print_detailed_prediction(
                predictions, image_path, top_k=len(class_names))
        else:
            print(f"Error al procesar: {image_path}")
