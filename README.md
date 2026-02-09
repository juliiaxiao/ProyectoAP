# ProyectoAP
**Asignatura:** Aprendizaje Profundo  
**Autores:** Jose García Mora y Júlia Xiao Moreno Delaplace

## 1. Descripción del Problema
El objetivo de este proyecto es desarrollar un modelo de Deep Learning capaz de clasificar imágenes de ropa en 10 categorías distintas. El dataset utilizado es **Fashion-MNIST**, que consta de 70,000 imágenes en escala de grises de $28 \times 28$ píxeles.

Es un problema de **aprendizaje supervisado** (clasificación multiclase) que sustituye al clásico MNIST de dígitos para ofrecer un reto ligeramente más complejo y realista.

## 2. Estado del Arte
A continuación se presenta una comparativa de modelos utilizados para resolver este problema y sus resultados:

| Modelo | Técnica / Arquitectura | Accuracy | Fuente |
| :--- | :--- | :--- | :--- |
| **Baseline (MLP)** | Red densa simple (3 capas) | ~88% | Zalando Research |
| **CNN Estándar** | Red Convolucional (2-3 capas) | ~92-93% | Keras Documentation |
| **ResNet-18** | Red Residual (Deep Learning) | **94.9%** | Benchmarks públicos |
| **WideResNet-28** | Arquitectura avanzada | **96.7%** | Estado del arte (SOTA) |

## 3. Métricas de Evaluación
Para medir el rendimiento de nuestros modelos utilizaremos:
* **Accuracy:** Para medir el porcentaje total de aciertos.
* **Matriz de Confusión:** Para identificar qué prendas se confunden más entre sí (ej. Camisas vs. Camisetas).
* **F1-Score:** Para asegurar un equilibrio entre precisión y exhaustividad en cada categoría.

## 4. Estructura del Proyecto
* `notebooks/`: Contiene el análisis exploratorio (EDA) y los futuros modelos.
* `data/`: Información sobre la fuente de datos.
* `requirements.txt`: Librerías necesarias para ejecutar el proyecto.

## 5. Instrucciones de Ejecución
El notebook principal `EDA_y_Carga.ipynb` está diseñado para ejecutarse directamente en **Google Colab**.
