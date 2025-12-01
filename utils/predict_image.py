import torch
import torch.nn.functional as F
from PIL import Image
import io
import os
import random
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Widgets globales (definidos aquí para que estén disponibles)
upload_widget = widgets.FileUpload(
    accept='.jpg,.jpeg,.png,.bmp,.tiff',
    multiple=False,
    description='📁 Seleccionar imagen'
)

output = widgets.Output()


def predict_single_image(model, image_path, class_names, transform, device, img_bytes=None):
    """
    Predice la clase de una imagen individual y muestra las probabilidades

    Args:
        model: Modelo PyTorch entrenado
        image_path: Ruta a la imagen (si viene desde disco)
        class_names: Lista de nombres de las clases
        transform: Transformaciones a aplicar a la imagen
        device: Dispositivo (CPU o GPU)
        img_bytes: Imagen en bytes (si viene desde el widget FileUpload)

    Returns:
        predictions: Lista de tuplas (clase, probabilidad)
        predicted_class: Clase con mayor probabilidad
        original_image: Imagen PIL original
    """
    try:
        # Permitir cargar por ruta o por bytes
        if img_bytes is not None:
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')

        original_image = image.copy()

        # Aplicar transformaciones
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Realizar predicción
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            # Aplicar softmax para obtener probabilidades
            probabilities = F.softmax(outputs, dim=1)

            # Obtener todas las probabilidades
            probs = probabilities.cpu().numpy()[0]

            # Crear lista de predicciones (clase, probabilidad)
            predictions = [(class_names[i], prob * 100)
                           for i, prob in enumerate(probs)]

            # Ordenar por probabilidad descendente
            predictions.sort(key=lambda x: x[1], reverse=True)

            # Obtener la clase con mayor probabilidad
            predicted_class = predictions[0][0]

        return predictions, predicted_class, original_image

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None, None, None


def visualize_prediction(image, predictions, predicted_class, image_path, top_k=5):
    """
    Visualiza la imagen y las predicciones de forma estética
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Mostrar imagen original
    ax1.imshow(image)
    ax1.set_title(f'Imagen: {os.path.basename(image_path)}\nPredicción: {predicted_class}',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Mostrar gráfico de barras con probabilidades
    if top_k:
        top_predictions = predictions[:top_k]
    else:
        top_predictions = predictions

    classes = [pred[0] for pred in top_predictions]
    probs = [pred[1] for pred in top_predictions]

    # Colores para las barras (la primera más destacada)
    colors = ['#2E86AB' if i == 0 else '#A23B72' for i in range(len(classes))]

    bars = ax2.barh(range(len(classes)), probs, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes)
    ax2.set_xlabel('Probabilidad (%)', fontsize=12)
    ax2.set_title(f'Top {len(top_predictions)} Predicciones',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Agregar valores en las barras
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 1, bar.get_y() + bar.get_height()/2,
                 f'{prob:.1f}%', va='center', fontweight='bold' if i == 0 else 'normal')

    plt.tight_layout()
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


def predict_from_folder(model, folder_path, class_names, transform, device, num_samples=5, top_k=3):
    """ 
    Selecciona imágenes aleatorias de una carpeta y realiza predicciones 
    """
    # Extensiones de imagen válidas
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    # Obtener todas las imágenes de la carpeta
    image_files = []
    for file in os.listdir(folder_path):
        if os.path.splitext(file.lower())[1] in valid_extensions:
            image_files.append(os.path.join(folder_path, file))
    if not image_files:
        print("No se encontraron imágenes en la carpeta especificada.")
        return
    # Seleccionar imágenes aleatorias
    selected_images = random.sample(
        image_files, min(num_samples, len(image_files)))
    print(
        f"\nAnalizando {len(selected_images)} imágenes de la carpeta: {folder_path}")
    for i, image_path in enumerate(selected_images):
        print(f"\n{'='*20} IMAGEN {i+1}/{len(selected_images)} {'='*20}")
        predictions, predicted_class, original_image = predict_single_image(
            model, image_path, class_names, transform, device
        )
        if predictions:
            visualize_prediction(original_image, predictions,
                                 predicted_class, image_path, top_k)
            print_prediction_summary(predictions, image_path, top_k)


def predecir_desde_widget(change, model, device, train_dataset, val_test_transform):
    """
    Función para manejar el widget de subida de archivos y realizar predicciones
    """
    output.clear_output()

    if not upload_widget.value:
        return

    # Obtener archivo subido (compatible con diferentes versiones de ipywidgets)
    try:
        # Para ipywidgets < 8.0
        file_info = next(iter(upload_widget.value.values()))
        file_bytes = file_info['content']
    except AttributeError:
        # Para ipywidgets >= 8.0
        if isinstance(upload_widget.value, tuple):
            file_info = upload_widget.value[0]
        else:
            file_info = list(upload_widget.value.values())[0]
        file_bytes = file_info['content']
        # Si 'content' es memoryview, convertir a bytes
        if isinstance(file_bytes, memoryview):
            file_bytes = file_bytes.tobytes()

    # Procesar la imagen
    predictions, predicted_class, original_image = predict_single_image(
        model=model,
        image_path=None,  # No usamos ruta, viene en bytes
        img_bytes=file_bytes,
        class_names=train_dataset.classes,
        transform=val_test_transform,
        device=device
    )

    # Mostrar resultados
    with output:
        print("\n🔍 Predicción realizada\n")
        if predictions:
            visualize_prediction(original_image, predictions, predicted_class,
                                 "imagen_subida", top_k=5)
            print_prediction_summary(predictions, "imagen_subida")
        else:
            print("❌ Error al procesar la imagen subida.")

# ============================================================================
# EJEMPLOS DE USO (puedes ejecutar estos en el notebook)
# ============================================================================


def ejemplo_imagen_especifica(model, train_dataset, val_test_transform, device):
    """Ejemplo para predecir una imagen específica"""

    # Ruta a tu imagen (modifica esta ruta)
    image_path = "datos/test/presente_1/image_001.jpg"  # Cambia por tu ruta

    # Verificar si el archivo existe
    if not os.path.exists(image_path):
        print(
            f"La imagen {image_path} no existe. Por favor, verifica la ruta.")
        return

    print("Analizando imagen específica...")

    predictions, predicted_class, original_image = predict_single_image(
        model=model,
        image_path=image_path,
        class_names=train_dataset.classes,
        transform=val_test_transform,
        device=device
    )

    if predictions:
        visualize_prediction(original_image, predictions,
                             predicted_class, image_path)
        print_prediction_summary(predictions, image_path)


def ejemplo_imagenes_test(model, train_dataset, val_test_transform, device):
    """Ejemplo para analizar imágenes aleatorias del conjunto de test"""

    test_folder = "datos/test"

    # Obtener todas las subcarpetas (clases)
    class_folders = [f for f in os.listdir(test_folder)
                     if os.path.isdir(os.path.join(test_folder, f))]

    print("Carpetas de clases encontradas:", class_folders)

    # Seleccionar una clase aleatoria
    selected_class = random.choice(class_folders)
    class_path = os.path.join(test_folder, selected_class)

    print(f"\nAnalizando imágenes de la clase: {selected_class}")

    predict_from_folder(
        model=model,
        folder_path=class_path,
        class_names=train_dataset.classes,
        transform=val_test_transform,
        device=device,
        num_samples=3,
        top_k=5
    )


def predecir_imagen_interactiva(model, train_dataset, val_test_transform, device):
    """Función interactiva para que el usuario ingrese la ruta de la imagen"""

    print("\n" + "="*50)
    print("PREDICCIÓN INTERACTIVA DE IMÁGENES")
    print("="*50)

    while True:
        image_path = input(
            "\nIngresa la ruta de la imagen (o 'quit' para salir): ").strip()

        if image_path.lower() == 'quit':
            break

        if not os.path.exists(image_path):
            print("❌ La imagen no existe. Verifica la ruta.")
            continue

        predictions, predicted_class, original_image = predict_single_image(
            model=model,
            image_path=image_path,
            class_names=train_dataset.classes,
            transform=val_test_transform,
            device=device
        )

        if predictions:
            visualize_prediction(original_image, predictions,
                                 predicted_class, image_path)
            print_prediction_summary(predictions, image_path)
        else:
            print("❌ Error al procesar la imagen.")
