import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

try:
    import ipywidgets as widgets
    from IPython.display import display
    import io

    # Variables globales que se configurarán desde el notebook
    trained_model = None
    val_test_transform = None
    class_names = None
    device = None

    # Crear widget de subida de archivos
    upload_widget = widgets.FileUpload(
        accept='.jpg,.jpeg,.png,.bmp,.tiff',
        multiple=False,
        description='📁 Seleccionar imagen'
    )

    output_widget = widgets.Output()

    def setup_widget_prediction(model, transform, classes, device_type):
        """
        Configura las variables necesarias para el widget de predicción

        Args:
            model: Modelo entrenado
            transform: Transformaciones para la imagen
            classes: Lista de nombres de clases
            device_type: Dispositivo de computación
        """
        global trained_model, val_test_transform, class_names, device
        trained_model = model
        val_test_transform = transform
        class_names = classes
        device = device_type
        print("✅ Widget de predicción configurado correctamente!")

    def predict_from_widget(change):
        """
        Función que se ejecuta cuando se selecciona una imagen
        """
        output_widget.clear_output()

        # Verificar que el widget esté configurado
        if trained_model is None or val_test_transform is None or class_names is None:
            with output_widget:
                print(
                    "❌ Error: Widget no configurado. Ejecuta setup_widget_prediction() primero.")
            return

        if not upload_widget.value:
            return

        with output_widget:
            try:
                # Obtener archivo subido (compatible con diferentes versiones de ipywidgets)
                try:
                    # Para ipywidgets < 8.0
                    file_info = next(iter(upload_widget.value.values()))
                    file_bytes = file_info['content']
                    filename = file_info['name']
                except (AttributeError, TypeError):
                    # Para ipywidgets >= 8.0
                    if isinstance(upload_widget.value, tuple) and len(upload_widget.value) > 0:
                        file_info = upload_widget.value[0]
                        file_bytes = file_info['content']
                        filename = file_info['name']
                    elif hasattr(upload_widget.value, 'values') and upload_widget.value.values():
                        file_info = list(upload_widget.value.values())[0]
                        file_bytes = file_info['content']
                        filename = file_info['name']
                    else:
                        print("❌ No se pudo obtener la imagen subida")
                        return

                # Si 'content' es memoryview, convertir a bytes
                if isinstance(file_bytes, memoryview):
                    file_bytes = file_bytes.tobytes()

                print(f"🔍 Analizando imagen: {filename}")
                print("-" * 50)

                # Cargar imagen desde bytes
                image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
                original_image = image.copy()

                # Aplicar transformaciones
                input_tensor = val_test_transform(
                    image).unsqueeze(0).to(device)

                # Realizar predicción
                trained_model.eval()
                with torch.no_grad():
                    outputs = trained_model(input_tensor)
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

                print(f"✅ Predicción completada!")
                print(f"Clase predicha: {predicted_class}")
                print(f"Confianza: {confidence:.2f}%")

                # Mostrar visualización
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Mostrar imagen original
                ax1.imshow(original_image)
                ax1.set_title(f'Imagen: {filename}\nPredicción: {predicted_class}',
                              fontsize=14, fontweight='bold')
                ax1.axis('off')

                # Mostrar gráfico de barras con probabilidades
                top_predictions = predictions[:len(class_names)]
                classes_display = [pred[0] for pred in top_predictions]
                probs_display = [pred[1] for pred in top_predictions]

                # Colores para las barras
                colors = ['#2E86AB' if i == 0 else '#A23B72' for i in range(
                    len(classes_display))]

                bars = ax2.barh(range(len(classes_display)),
                                probs_display, color=colors, alpha=0.8)
                ax2.set_yticks(range(len(classes_display)))
                ax2.set_yticklabels(classes_display, fontsize=12)
                ax2.set_xlabel('Probabilidad (%)', fontsize=12)
                ax2.set_title('Predicciones por Clase',
                              fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)

                # Agregar valores en las barras
                for i, (bar, prob) in enumerate(zip(bars, probs_display)):
                    ax2.text(prob + 1, bar.get_y() + bar.get_height()/2,
                             f'{prob:.1f}%', va='center', ha='left',
                             fontweight='bold' if i == 0 else 'normal')

                plt.tight_layout()
                plt.show()

                # Imprimir resumen detallado
                print("\n" + "="*60)
                print(f"RESUMEN DE PREDICCIÓN PARA: {filename}")
                print("="*60)

                for i, (class_name, probability) in enumerate(predictions):
                    status = "⭐ PREDICCIÓN PRINCIPAL" if i == 0 else ""
                    print(
                        f"{i+1:2d}. {class_name:<15} {probability:6.2f}% {status}")

                print("="*60)
                print("✅ Análisis completado exitosamente!")

            except Exception as e:
                print(f"❌ Error al procesar la imagen: {e}")
                print("Verifica que el archivo sea una imagen válida (JPG, PNG, etc.)")
                import traceback
                traceback.print_exc()

    def display_prediction_widget():
        """
        Muestra el widget de predicción con instrucciones
        """
        if trained_model is None:
            print("❌ Error: Primero configura el widget con setup_widget_prediction()")
            return

        print("📋 Instrucciones:")
        print("1. Haz clic en 'Seleccionar imagen' abajo")
        print("2. Selecciona una imagen desde tu computadora")
        print("3. La predicción se mostrará automáticamente")
        print("\nFormatos soportados: JPG, JPEG, PNG, BMP, TIFF")
        print("-" * 60)

        display(upload_widget)
        display(output_widget)

    # Conectar la función al widget
    upload_widget.observe(predict_from_widget, names='value')

except ImportError as e:
    print("❌ ipywidgets no está instalado.")
    print("Para usar esta funcionalidad, instala ipywidgets:")
    print("pip install ipywidgets")
    print("\nO usa las opciones 1 y 2 de arriba para predicciones sin widgets.")

    # Definir funciones dummy para evitar errores de importación
    def setup_widget_prediction(*args):
        print("❌ ipywidgets no disponible")

    def display_prediction_widget():
        print("❌ ipywidgets no disponible")

    upload_widget = None
    output_widget = None
    predict_from_widget = None
