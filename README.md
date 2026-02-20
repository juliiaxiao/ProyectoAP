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
El notebook principal `EDA.ipynb` está diseñado para ejecutarse directamente en **Google Colab**.

## 6. Cómo hacer un Pull (Actualizar tu copia local)

Para descargar los últimos cambios del repositorio remoto a tu copia local, sigue estos pasos:

1. **Abre una terminal** en la carpeta del proyecto (o en Google Colab usa `!` antes del comando).

2. **Asegúrate de estar en la rama correcta:**
   ```bash
   git status
   ```

3. **Descarga los cambios remotos:**
   ```bash
   git pull origin main
   ```
   > Si la rama principal se llama `master`, usa `git pull origin master`.

4. **Flujo de trabajo habitual con Git:**
   ```bash
   # 1. Ver el estado de tus archivos
   git status

   # 2. Añadir los cambios que quieres confirmar
   git add .

   # 3. Confirmar los cambios con un mensaje descriptivo
   git commit -m "Descripción de los cambios"

   # 4. Subir los cambios al repositorio remoto
   git push origin main

   # 5. Descargar los últimos cambios del repositorio remoto
   git pull origin main
   ```

> **Nota:** Si tienes cambios locales sin confirmar y hay cambios remotos, Git puede pedir que primero hagas `commit` o `stash` de tus cambios antes de hacer `pull`.
