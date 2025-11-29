# 📊 Plantilla de Presentación - Transfer Learning

**Proyecto 3: Transfer Learning para Clasificación de Frutas y Verduras**  
**INFO1185 - Inteligencia Artificial**  
**Prof. Dr. Ricardo Soto Catalán**  
**Noviembre 2025**

---

## 🎯 Slide 1: Portada

### Transfer Learning para Clasificación de Frutas y Verduras
**Estudiantes**: [Nombre 1] y [Nombre 2]  
**Curso**: INFO1185 - Inteligencia Artificial  
**Profesor**: Prof. Dr. Ricardo Soto Catalán  
**Fecha**: 03 de diciembre de 2025  

**Modelo seleccionado**: [ResNet18 / VGG16 / DenseNet121 / etc.]

---

## 📋 Slide 2: Descripción del Problema (5 puntos)

### 🎯 Objetivo del Proyecto
- **Problema**: Clasificación automática de frutas y verduras en imágenes
- **Técnica**: Transfer Learning con modelos preentrenados
- **Desafío**: Adaptar modelos de ImageNet a nuestro dominio específico

### 🔍 Motivación
- **Aplicaciones reales**: Sistemas de inventario, clasificación automática, agricultura de precisión
- **Beneficios del Transfer Learning**: Menor costo computacional, aprovecha conocimiento previo
- **Comparación**: Evaluar diferentes arquitecturas de clasificador

### 📌 Objetivos Específicos
1. Implementar Transfer Learning con [Modelo elegido]
2. Comparar clasificador simple vs. embudo
3. Analizar impacto de técnicas de regularización
4. Evaluar rendimiento con métricas estándar

---

## 📊 Slide 3: Descripción del Dataset (5 puntos)

### 📁 Fruits and Vegetables Image Recognition Dataset

#### 📈 Estadísticas del Dataset
- **Fuente**: Kaggle - Muhammad Ehsan
- **Total de imágenes**: [X,XXX] imágenes
- **Número de clases**: [XX] clases diferentes
- **Resolución**: Variada, redimensionada a 224×224 píxeles

#### 🗂️ División del Dataset
| Conjunto | Cantidad | Porcentaje |
|----------|----------|------------|
| **Entrenamiento** | [X,XXX] | 70% |
| **Validación** | [XXX] | 20% |
| **Prueba** | [XXX] | 10% |

#### 🏷️ Clases Incluidas
- **Frutas**: [Manzana, Banana, Naranja, ...]
- **Verduras**: [Zanahoria, Tomate, Brócoli, ...]

*[Mostrar gráfico de distribución de clases y muestras del dataset]*

---

## 🧠 Slide 4: Arquitectura del Modelo (10 puntos)

### 🏗️ Modelo Base: [Nombre del modelo seleccionado]

#### 🔧 Configuración del Backbone
- **Modelo preentrenado**: [ResNet18/VGG16/etc.] entrenado en ImageNet
- **Congelamiento**: Todas las capas del backbone congeladas
- **Extractor de características**: [Tamaño de salida] características

#### 🏛️ Versión 1: Clasificador Simple
```
Backbone → FC([tamaño_características] → [num_clases])
```
- ✅ Una única capa totalmente conectada
- ❌ Sin Batch Normalization
- ❌ Sin Dropout

#### 🏛️ Versión 2: Clasificador Embudo
```
Backbone → FC(512) → BN → ReLU → Dropout → 
           FC(256) → BN → ReLU → Dropout → 
           FC([num_clases])
```
- ✅ Arquitectura tipo embudo (512 → 256 → clases)
- ✅ Batch Normalization entre capas
- ✅ Dropout (p=0.3) para regularización
- ✅ Activaciones ReLU

---

## ⚙️ Slide 5: Estrategia de Entrenamiento (10 puntos)

### 🎯 Configuración del Entrenamiento

#### 📊 Hiperparámetros
| Parámetro | Valor |
|-----------|--------|
| **Learning Rate** | 0.001 |
| **Batch Size** | 32 |
| **Optimizador** | Adam |
| **Weight Decay** | 1e-4 |
| **Épocas máximas** | 50 |

#### 🛡️ Técnicas de Regularización
- **Early Stopping**: Paciencia de 10 épocas
- **Data Augmentation** (solo entrenamiento):
  - Flip horizontal aleatorio
  - Rotación ±15°
  - ColorJitter (brillo, contraste, saturación)
  - Random Resized Crop

#### 🔄 Experimentos Realizados
1. **V1**: Clasificador simple
2. **V2 sin regularización**: Embudo sin BN ni Dropout  
3. **V2 con regularización**: Embudo completo con BN + Dropout

---

## 📈 Slide 6: Resultados Obtenidos (10 puntos)

### 🏆 Comparación de Rendimiento

| Modelo | Precisión Test | Pérdida Test | Parámetros Entrenables |
|--------|----------------|--------------|------------------------|
| **V1 Simple** | [XX.X%] | [0.XXX] | [XXX] |
| **V2 Sin Reg** | [XX.X%] | [0.XXX] | [X,XXX] |
| **V2 Con Reg** | [XX.X%] | [0.XXX] | [X,XXX] |

### 📊 Métricas Detalladas por Clase
*[Mostrar tabla con Precision, Recall, F1-Score para cada clase]*

### 🎨 Visualizaciones Clave
1. **Curvas de entrenamiento**: Pérdida y precisión vs épocas
2. **Matriz de confusión**: Para el mejor modelo
3. **Gráfico comparativo**: Barras de precisión por modelo

*[Insertar gráficos reales aquí]*

### 🔍 Observaciones Principales
- Modelo con mejor rendimiento: **[Nombre]**
- Mejora de V1 a V2: **[+X.X%]**
- Impacto de regularización: **[+/-X.X%]**

---

## 🔬 Slide 7: Análisis y Discusión (10 puntos)

### 📊 Impacto de la Arquitectura
- **V1 vs V2**: El clasificador embudo mostró [mejora/similar/peor] rendimiento
- **Capacidad de representación**: Mayor número de parámetros permitió [mejor/similar] captura de patrones
- **Complejidad vs Rendimiento**: [Análisis del trade-off]

### 🛡️ Efecto de las Técnicas de Regularización
- **Batch Normalization**: [Impacto observado en estabilidad/convergencia]
- **Dropout**: [Efecto en sobreajuste/generalización]
- **Combinación BN + Dropout**: [Resultado sinérgico/individual]

### ⚡ Estabilidad del Entrenamiento
- **Convergencia**: [Análisis de las curvas de pérdida]
- **Overfitting**: [Observaciones sobre la diferencia train-val]
- **Early Stopping**: Activado en [X] de [Y] experimentos

### 💻 Limitaciones de Google Colab
- **Memoria GPU**: [Limitaciones encontradas]
- **Tiempo de entrenamiento**: [Restricciones temporales]
- **Soluciones aplicadas**: [Batch size reducido, modelos más pequeños, etc.]

---

## 🏁 Slide 8: Conclusiones (10 puntos)

### ✅ Logros Principales
1. **Implementación exitosa** de Transfer Learning con [modelo]
2. **Comparación exhaustiva** entre arquitecturas simples y complejas
3. **Análisis detallado** del impacto de técnicas de regularización
4. **Evaluación completa** con métricas estándar de clasificación

### 📊 Hallazgos Clave
- **Mejor modelo**: [Nombre] con [XX.X%] de precisión
- **Arquitectura óptima**: [Simple/Embudo] según nuestros datos
- **Regularización**: [Beneficial/No beneficial] en nuestro caso
- **Transfer Learning**: Efectivo para clasificación de frutas/verduras

### 🔮 Trabajo Futuro
- **Modelos avanzados**: Probar EfficientNet, Vision Transformers
- **Fine-tuning**: Descongelar capas finales del backbone
- **Data Augmentation**: Técnicas más sofisticadas (MixUp, CutMix)
- **Ensemble methods**: Combinar múltiples modelos

### 🎯 Lecciones Aprendidas
- Importancia del **balance** entre complejidad y datos disponibles
- **Regularización** como herramienta clave para generalización
- **Monitoreo continuo** necesario para evitar overfitting
- **Transfer Learning** como estrategia efectiva para dominios específicos

---

## ❓ Slide 9: Preguntas Frecuentes y Discusión

### 🤔 Posibles Preguntas del Profesor/Audiencia

**P1: ¿Por qué eligieron [modelo específico]?**
- R: [Razones técnicas: tamaño, rendimiento, recursos disponibles]

**P2: ¿Cómo manejaron el desbalance de clases?**
- R: [Estrategias aplicadas o por qué no fue necesario]

**P3: ¿Qué pasaría si descongelaran capas del backbone?**
- R: [Análisis teórico basado en literatura y recursos disponibles]

**P4: ¿Cómo validaron que no hay data leakage?**
- R: [Explicar división de datasets y validación cruzada]

**P5: ¿Cuál sería el siguiente paso para mejorar resultados?**
- R: [Propuestas concretas y justificadas]

### 🎯 Preparación para Demo
- **Código ejecutable** listo para mostrar
- **Modelos entrenados** guardados y disponibles
- **Visualizaciones** preparadas para explicar resultados
- **Métricas** calculadas y listas para discutir

---

## 📋 Checklist de Presentación

### ✅ Antes de Presentar
- [ ] Slides revisados y sin errores
- [ ] Tiempos ensayados (máx 8 minutos)
- [ ] Código funcionando correctamente
- [ ] Resultados reales incluidos
- [ ] Respuestas a preguntas preparadas
- [ ] Participación equilibrada del equipo

### 📊 Elementos Visuales Requeridos
- [ ] Gráfico de distribución del dataset
- [ ] Curvas de entrenamiento (pérdida y precisión)
- [ ] Matriz de confusión del mejor modelo
- [ ] Comparación entre modelos (gráfico de barras)
- [ ] Ejemplos de imágenes clasificadas

### 🎯 Puntos de la Rúbrica Cubiertos
- [ ] Descripción clara del problema (5 pts)
- [ ] Dataset explicado con números (5 pts)
- [ ] Arquitecturas justificadas técnicamente (10 pts)
- [ ] Resultados con métricas completas (10 pts)
- [ ] Conclusiones sólidas y conectadas (10 pts)
- [ ] Comunicación fluida y tiempo respetado (10 pts)
- [ ] Preparación para preguntas (10 pts)

---

**🕒 Duración total**: 8 minutos máximo  
**🎤 Fecha de presentación**: 03 de diciembre, 13:50 hrs  
**📝 Entrega del código**: 03 de diciembre, 13:00 hrs

### 💡 Consejos Finales
- **Practiquen** la presentación múltiples veces
- **Cronometren** cada sección para no exceder 8 minutos  
- **Preparen** respuestas para preguntas técnicas comunes
- **Aseguren** participación equilibrada entre integrantes
- **Tengan** backup del código y slides
