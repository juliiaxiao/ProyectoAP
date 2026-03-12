import tensorflow as tf
from tensorflow.keras import layers, models

def get_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        
        # Cross entropy estándar
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Probabilidad predicha de la clase correcta
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        
        # Focal Loss
        focal_factor = alpha * tf.pow(1.0 - p_t, gamma)
        loss = focal_factor * ce
        
        return loss
    
    return focal_loss
"""
def create_fashion_cnn(input_shape=(28, 28, 1), num_classes=10):

    model = models.Sequential([

        # Bloque 1: Captura bordes y texturas simples
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        #layers.Dropout(0.2), # Dropout estándar aquí está bien

        # Bloque 2: Empezamos a detectar formas de prendas
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # Agregamos SpatialDropout2D para que no dependa de un solo mapa de características
        layers.SpatialDropout2D(0.1), 
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        # Bloque 3: Características complejas (cuellos, mangas, botones)
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        # Mayor regularización espacial aquí
        layers.SpatialDropout2D(0.2), 
        layers.MaxPooling2D((2,2)),
        #layers.Dropout(0.2),

        # Clasificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Aumentamos ligeramente el dropout final
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
"""
def create_fashion_cnn_v3(input_shape=(28, 28, 3), num_classes=10):
    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        # Bloque 2 - Aplicamos SpatialDropout para forzar el uso de los 3 canales
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2), 
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Bloque 3 - Extracción profunda
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.2),
        layers.MaxPooling2D((2,2)),

        # Clasificador
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'), # Aumentamos neuronas para procesar los nuevos canales
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


    