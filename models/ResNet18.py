import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_block(inputs, filters, strides=1, l2=1e-4):
    x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2))(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs
        
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def create_resnet18(input_shape=(28, 28, 1), num_classes=10, l2=1e-4, dropout=0.3):
    inputs = layers.Input(shape=input_shape)
    
    # Capa inicial
    # Usamos kernel=3 y strides=1 en vez de kernel=7, strides=2 porque la imagen de entrada es de 28x28 (pequeña)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Max pooling omitido dado el pequeño tamaño inicial de las imágenes
    
    # Layer 1
    x = resnet_block(x, 64, l2=l2)
    x = resnet_block(x, 64, l2=l2)
    
    # Layer 2
    x = resnet_block(x, 128, strides=2, l2=l2)
    x = resnet_block(x, 128, l2=l2)
    
    # Layer 3
    x = resnet_block(x, 256, strides=2, l2=l2)
    x = resnet_block(x, 256, l2=l2)
    
    # Layer 4
    #x = resnet_block(x, 512, strides=2, l2=l2)
    #x = resnet_block(x, 512, l2=l2)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    
    model = models.Model(inputs, outputs)
    return model

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
