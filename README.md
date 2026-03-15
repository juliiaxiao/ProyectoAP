# ProyectoAP
**Asignatura:** Aprendizaje Profundo  
**Autores:** Jose García Mora y Júlia Xiao Moreno Delaplace

## 1. Descripción del Problema
El objetivo de este proyecto es desarrollar un modelo de Deep Learning capaz de clasificar imágenes de ropa en 10 categorías distintas. El dataset utilizado es **Fashion-MNIST**, que consta de 70,000 imágenes, 60,000 para entrenamiento y 10,000 para test, en escala de grises de $28 \times 28$ píxeles. La salida serán etiqueta de números enteros del 0 al 9.

Es un problema de **aprendizaje supervisado** (clasificación multiclase) que sustituye al clásico MNIST de dígitos para ofrecer un reto ligeramente más complejo y realista.

### Ejemplo del Dato (Figura 1):
![Ejemplos de Fashion-MNIST](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

*Figura 1: Infografía de los datos. De la entrada (píxeles) se espera una salida clasificatoria (ej: 9=Botín, 0=Camiseta).*

## 2. Estado del Arte
A continuación se presenta una comparativa de modelos utilizados para resolver este problema y sus resultados:

| Modelo | Técnica / Arquitectura | Accuracy | Fuente |
| :--- | :--- | :--- | :--- |
| **Baseline (MLP)** | Red densa simple (3 capas) | ~88% | Zalando Research |
| **CNN Estándar** | CNN simple (2–3 capas convolucionales) | ~92–93% | Keras Documentation |
| **CNN + BN + Dropout** | CNN regularizada | ~94–95% | Implementaciones open-source |
| **MobileNetV2** | Convoluciones separables en profundidad | **95–96%** | TensorFlow / PyTorch (Apache 2.0) |


## 3. Métricas de Evaluación
Para medir el rendimiento de nuestros modelos utilizaremos:
* **Accuracy:** Para medir el porcentaje total de aciertos.
* **Matriz de Confusión:** Para identificar qué prendas se confunden más entre sí (ej. Camisas vs. Camisetas).
* **F1-Score:** Para asegurar un equilibrio entre precisión y exhaustividad en cada categoría.

## 4. Estructura del Proyecto
* `notebooks/`: Contiene el análisis exploratorio (EDA) y los futuros modelos.
* `requirements.txt`: Librerías necesarias para ejecutar el proyecto.

## 5. Instrucciones de Ejecución
Los notebooks está diseñados para ejecutarse directamente en **Google Colab**.

## 6. Modelos Simples

| Modelo | Parámetros	| Train Acc |	Val Acc	|Test Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Modelo Lineal (Softmax)** | 7,850 | 0.8684 | 0.8537 |0.8430|
| **Machine Learning (SVM RBF C=10)** | 15,760 | 0.9736 | 0.9038 | 0.8978 |
| **Machine Learning (SVM RBF C=10 con HOG)** | 25,593,408 | 0.9999 | 0.9129 |0.9099|
| **Machine Learning (SVM RBF C=1 con HOG)** | 25,593,408 |0.9500 | 0.9072 |0.9035|
| **Red Neuronal (CNN)** | 1,726 | 0.8490 | 0.8454 | 0.8395 |

- **Fase 1 (Modelo Lineal):** Establecimos nuestro baseline inicial alimentando los píxeless a una regresión Softmax. Con 7,850 parámetros logramos un  84.30% en Test. La brecha entre Entrenamiento y Test es mínima (~2.5%), lo que demuestra que no hay overfitting, pero evidencia que hemos tocado el "techo" de aprendizaje de las fronteras lineales.
- **Fase 2 (Machine Learning Clásico):** Al introducir descriptores espaciales (HOG) y un Kernel RBF no lineal, logramos romper la barrera del 90% de acierto. Sin embargo, esto demostró dos problemas: un gran overfitting inicial ($99.99\%$ en Train con $C=10$, mitigado al relajar el margen a $C=1$) y un coste computacional demasiado grande, requiriendo más de 25.5 millones de parámetros efectivos basados en sus vectores de soporte.
- **Fase 3 (Deep Learning):** Buscando la arquitectura "más simple posible", diseñamos una Red Neuronal Convolucional (CNN) extremadamente ligera. Con tan solo 1,726 parámetros, el modelo es capaz de rozar el 84% en Test. Muestra una generalización casi perfecta, demostrando una eficiencia paramétrica inmensamente superior al Machine Learning clásico.

## 7. Modelos Complejos

| Modelo | Parámetros	| Train Acc |	Val Acc	|Test Acc |
| :--- | :--- | :--- | :--- | :--- |
| **Transformer Autoencoder** | 354,699** | 0.8045 | 0.7970 | 0.8030 |
| **Transformer Minimal** | 27,770** | 0.7480 | 0.7430 | 0.7320 |
| **Attention Minimal** | 805 | 0.3871 | 0.3610 | 0.3605** |
| **Attention CNN (CBAM)** | 168,479 | 0.0972 | 0.1060 | 0.0985 |
| **ResNet18** | ~2,781,706 | 1.0000 | 0.9296 | 0.9270 |
| **CNN con Atención** | 74,986 |  0.9140 | 0.9262 |  0.9173 |


- **Fase 4 (Arquitecturas Transformer):** Experimentamos con el paradigma de atención global mediante un Transformer Autoencoder. Con 354,699 parámetros logramos un 80.30% en Test. Aunque supera la barrera del modelo lineal, la eficiencia es menor que en las CNNs para este dataset de baja resolución. Al reducir el modelo a un Transformer Minimal (27,770 parámetros), el acierto cayó al 73.20%, confirmando que estas arquitecturas requieren una mayor profundidad paramétrica para aprender representaciones útiles en Fashion-MNIST.
- **Fase 5 (Atención Pura):** Buscando el límite inferior de complejidad, diseñamos un mecanismo de Attention Minimal con apenas 805 parámetros. El pobre rendimiento obtenido (36.05% en Test) demuestra que la atención por sí sola, sin una base convolucional previa que extraiga características espaciales, es insuficiente para distinguir la morfología de las prendas.
- **Fase 6 (Redes Residuales Profundas):** Implementamos una ResNet18 para buscar el techo de precisión del proyecto. Logramos nuestro mejor resultado bruto con un 92.70% en Test, pero a un coste de 2,781,002 parámetros y con un evidente overfitting (100% en Train). A pesar de usar Focal Loss , la arquitectura resulta excesivamente compleja para la tarea, penalizando la eficiencia que buscamos en la asignatura.
- **Fase 7 (Modelo Final - CNN con atención):** Nuestra solución definitiva integra un bloque de atención CBAM sobre una base convolucional ligera. Con solo 74,986 parámetros, alcanzamos un 91.73% en Test, logrando el equilibrio óptimo entre ligereza y capacidad predictiva. La mínima brecha entre Validación (92.62%) y Test demuestra una generalización robusta, validando nuestra estrategia de Data Augmentation y ajuste de hiperparámetros.

## Análisis de la Cota de Acierto y Calidad del Dato

Se ha observado que el rendimiento de los modelos (especialmente en la CNN con CBAM) tiende a estabilizarse en torno al 92% de accuracy en test. Tras un análisis profundo en el EDA, atribuimos esta limitación a dos factores críticos:
- **Baja densidad de información:** Clases como Sandal o Ankle boot presentan una alta proporción de píxeles nulos (fondo), lo que dificulta la extracción de características en una resolución de $28 \times 28$.
- **Ambigüedad Visual:** Hay ejemplares donde incluso para el ojo humano es imposible distinguir el tipo de prenda, estableciendo un techo logístico de acierto que no se puede superar sin caer en el overfitting de ruido.
- **Solapamiento Inter-clase:** Existe un solapamiento estadístico significativo entre las clases Shirt, T-shirt/top y Coat. Forzar un acierto superior mediante arquitecturas más complejas (como la ResNet18) solo conduce a un escenario de sobreajuste ($Accuracy_{train} = 1.0$) sin una mejora real en la capacidad de generalización sobre la función de pérdida:

 $$\mathcal{L} = -\sum_{i=1}^{M} y_i \log(\hat{y}_i)$$

 
Por tanto, consideramos que nuestro modelo de 74,986 parámetros representa el equilibrio entre la solución más refinada para el dataset y la maximización del acierto por cada parámetro invertido.

