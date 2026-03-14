import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def cbam_block(cbam_feature, ratio=8):
    """Módulo de Atención Convolucional"""
    channel_axis = -1
    channel = cbam_feature.shape[channel_axis]
    
    # Channel Attention
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_layer_two = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)
    
    avg_pool = layers.GlobalAveragePooling2D()(cbam_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D()(cbam_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature_channel = layers.Add()([avg_pool, max_pool])
    cbam_feature_channel = layers.Activation('sigmoid')(cbam_feature_channel)
    x = layers.Multiply()([cbam_feature, cbam_feature_channel])
    
    # Spatial Attention
    avg_pool_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(x)
    max_pool_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=3, keepdims=True))(x)
    concat = layers.Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])
    spatial_attention = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    
    return layers.Multiply()([x, spatial_attention])

def create_fashion_cnn_v5(input_shape=(28, 28, 3), num_classes=10):
    """
    ARQUITECTURA OPCIÓN 1: Alta resolución espacial para detalles finos.
    """
    inputs = layers.Input(shape=input_shape)

    # Bloque 1: Resolución 28x28
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2,2))(x) # Baja a 14x14
    x = layers.Dropout(0.1)(x)

    # Bloque 2: Resolución 14x14
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = cbam_block(x)
    x = layers.MaxPooling2D((2,2))(x) # Baja a 7x7
    x = layers.Dropout(0.1)(x)

    # Bloque 3: Mantenemos resolución 7x7 (eliminamos el pooling final)
    # Esto es CRUCIAL para que la atención detecte la estructura de la camisa
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout2D(0.1)(x)
    x = cbam_block(x)
    # <-- No hay MaxPooling aquí

    # En vuestro archivo .py, sección del Clasificador:
    x = layers.Flatten()(x) # En lugar de GlobalAveragePooling2D
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) # Subimos un poco el dropout al usar Flatten
    outputs = layers.Dense(num_classes, activation='softmax')(x)    

    model = models.Model(inputs=inputs, outputs=outputs)
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