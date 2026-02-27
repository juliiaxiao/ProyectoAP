import tensorflow as tf
from tensorflow.keras import layers, models
from keras.optimizers import Adam

def build_conv_minimal(input_shape=(28, 28, 1), num_classes=10):
    """
    Arquitectura CNN Evolucionada:
    - Dos capas convolucionales para ganar profundidad.
    - Global Average Pooling para mantener los parámetros bajos.
    """
    model = models.Sequential([
        # Capa 1: Extrae rasgos básicos (bordes)
        layers.Conv2D(filters=10, kernel_size=(3, 3), padding='same',
                      activation='relu', input_shape=input_shape),

        # Capa 2: Aporta mayor profundidad a la red
        layers.Conv2D(filters=16, kernel_size=(3, 3),
                      padding='same', activation='relu'),

        # Global Average Pooling
        layers.GlobalAveragePooling2D(),

        layers.Activation('softmax')
        #layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model