# -*- coding: utf-8 -*-
"""
Modelos con Capas de Atención para Fashion MNIST

Implementa:
1. Spatial Attention CNN - combina CNN con atención espacial
2. Channel Attention Network - atención en canales de feature maps  
3. CBAM (Convolutional Block Attention Module) - atención espacial + canal
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class SpatialAttention(layers.Layer):
    """
    Mecanismo de atención espacial que aprende dónde enfocar la atención
    en el espacio de la imagen (H x W)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # Capa convolucional para generar mapa de atención
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        # Promedio y máximo a través de los canales
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)  # (B, H, W, 1)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)   # (B, H, W, 1)
        
        # Concatenar estadísticas
        concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])  # (B, H, W, 2)
        
        # Generar mapa de atención
        attention_map = self.conv(concat)  # (B, H, W, 1)
        
        # Aplicar atención
        return inputs * attention_map


class ChannelAttention(layers.Layer):
    """
    Mecanismo de atención en canales - aprende qué canales/features 
    son más importantes
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(1, channels // self.reduction_ratio)
        
        # MLP compartido para avg y max pooling
        self.shared_mlp = tf.keras.Sequential([
            layers.Dense(reduced_channels, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])
        
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        
    def call(self, inputs):
        # Channel-wise pooling
        avg_pool = self.global_avg_pool(inputs)  # (B, 1, 1, C)
        max_pool = self.global_max_pool(inputs)  # (B, 1, 1, C)
        
        # Squeeze spatial dimensions
        avg_pool = tf.squeeze(avg_pool, axis=[1, 2])  # (B, C)
        max_pool = tf.squeeze(max_pool, axis=[1, 2])  # (B, C)
        
        # Aplicar MLP compartido
        avg_attention = self.shared_mlp(avg_pool)
        max_attention = self.shared_mlp(max_pool)
        
        # Combinar y expandir dimensiones
        attention_weights = avg_attention + max_attention
        attention_weights = tf.expand_dims(tf.expand_dims(attention_weights, 1), 1)
        
        # Aplicar atención
        return inputs * attention_weights


class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module
    Combina atención de canal y espacial secuencialmente
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channel_attention = ChannelAttention(reduction_ratio)
        self.spatial_attention = SpatialAttention()
        
    def call(self, inputs):
        # Primero atención de canal, luego espacial
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x


def create_attention_cnn(
    input_shape=(28, 28, 1),
    num_classes=10,
    use_cbam=True,
    base_filters=32
):
    """
    CNN con mecanismos de atención (CBAM)
    
    Args:
        input_shape: forma de entrada
        num_classes: número de clases
        use_cbam: usar CBAM (True) o solo spatial attention (False)
        base_filters: número base de filtros
    
    Returns:
        Modelo Keras compilado
        
    Parámetros aprox: ~95k (más eficiente que CNN óptima)
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # ── Bloque 1: Extracción de features básicas ───────────────────────────
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(base_filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Aplicar atención después del primer bloque
    if use_cbam:
        x = CBAM(reduction_ratio=8)(x)
    else:
        x = SpatialAttention()(x)
    
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # ── Bloque 2: Features más complejas ───────────────────────────────────
    x = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Atención en el segundo bloque
    if use_cbam:
        x = CBAM(reduction_ratio=8)(x)
    else:
        x = SpatialAttention()(x)
    
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # ── Bloque 3: Features de alto nivel ───────────────────────────────────
    x = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Atención final
    if use_cbam:
        x = CBAM(reduction_ratio=4)(x)
    else:
        x = SpatialAttention()(x)
    
    # ── Clasificador ────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Crear y compilar modelo
    model = Model(inputs, outputs, name='AttentionCNN')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_residual_attention_net(input_shape=(28, 28, 1), num_classes=10):
    """
    Red residual con módulos de atención
    Inspirada en ResNet pero con atención espacial
    """
    inputs = layers.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Bloque residual con atención
    def residual_attention_block(x, filters, stride=1):
        shortcut = x
        
        # Primera conv
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Segunda conv
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Módulo de atención
        x = SpatialAttention()(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    # Stack de bloques residuales con atención
    x = residual_attention_block(x, 32)
    x = residual_attention_block(x, 32)
    x = residual_attention_block(x, 64, stride=2)
    x = residual_attention_block(x, 64)
    x = residual_attention_block(x, 128, stride=2)
    
    # Clasificador
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResidualAttentionNet')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_attention_minimal(input_shape=(28, 28, 1), num_classes=10):
    """
    Modelo minimal con atención para competir con MLP minimal
    Objetivo: <8k parámetros con atención espacial
    """
    inputs = layers.Input(shape=input_shape)
    
    # Una sola capa conv con atención
    x = layers.Conv2D(16, 5, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Atención espacial simple
    x = SpatialAttention()(x)
    
    # Pooling adaptativo
    x = layers.GlobalAveragePooling2D()(x)
    
    # Clasificador minimal
    x = layers.Dense(8, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='AttentionMinimal')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Test del modelo
    model = create_attention_cnn()
    model.summary()
    print(f"\nParámetros totales: {model.count_params():,}")
    
    # Test atención minimal
    print("\n" + "="*50)
    model_min = create_attention_minimal()
    model_min.summary()
    print(f"\nParámetros totales (minimal): {model_min.count_params():,}")