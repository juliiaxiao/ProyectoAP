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

