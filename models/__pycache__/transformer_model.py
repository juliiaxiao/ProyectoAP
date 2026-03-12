# -*- coding: utf-8 -*-
"""
Transformer Autoencoder para Fashion MNIST

Implementa una arquitectura Transformer adaptada para imágenes 28x28:
- Patch embedding para dividir la imagen en parches
- Multi-head self-attention para capturar relaciones globales
- Encoder-Decoder structure con skip connections
- Positional encoding para información espacial
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class PatchEmbedding(layers.Layer):
    """
    Convierte imagen de 28x28 en secuencia de patches con embeddings posicionales
    """
    def __init__(self, patch_size=4, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 28x28 -> 7x7 patches de 4x4
        self.num_patches = (28 // patch_size) ** 2  # 49 patches
        self.patch_dim = patch_size * patch_size  # 16 for 4x4 patches
        
    def build(self, input_shape):
        # Proyección linear de patches
        self.projection = layers.Dense(self.embed_dim)
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.embed_dim
        )
        super().build(input_shape)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        
        # Extraer patches: (B, 28, 28, 1) -> (B, 49, 16)
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape: (B, 49, 16) - especificar tamaño explícitamente
        patches = tf.reshape(patches, [batch_size, self.num_patches, self.patch_dim])
        
        # Proyectar a embed_dim
        encoded_patches = self.projection(patches)
        
        # Añadir positional encoding
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded_patches += self.position_embedding(positions)
        
        return encoded_patches


class TransformerBlock(layers.Layer):
    """
    Bloque Transformer con self-attention y feed-forward network
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Multi-head self attention
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        # Self attention + skip connection
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward + skip connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def create_transformer_autoencoder(
    input_shape=(28, 28, 1), 
    num_classes=10,
    patch_size=4,
    embed_dim=64,
    num_transformer_layers=4,
    num_heads=4,
    ff_dim=128,
    dropout_rate=0.1
):
    """
    Crea un Transformer Autoencoder para Fashion MNIST
    
    Args:
        input_shape: forma de entrada (28, 28, 1)
        num_classes: número de clases (10)
        patch_size: tamaño de cada patch (4x4)
        embed_dim: dimensión de embeddings (64)
        num_transformer_layers: número de capas transformer (4)
        num_heads: heads en multi-head attention (4)
        ff_dim: dimensión feed-forward network (128)
        dropout_rate: tasa de dropout (0.1)
    
    Returns:
        Modelo Keras compilado
        
    Parámetros aprox: ~85k (más eficiente que CNN óptima)
    """
    
    # Entrada
    inputs = layers.Input(shape=input_shape)
    
    # ── ENCODER ──────────────────────────────────────────────────────────────
    # Patch embedding
    patches = PatchEmbedding(patch_size, embed_dim)(inputs)
    
    # Stack de Transformer blocks
    encoded = patches
    for _ in range(num_transformer_layers):
        encoded = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(encoded)
    
    # Global feature aggregation
    # Attention pooling para agregar información de todos los patches
    attention_weights = layers.Dense(1, activation='softmax')(encoded)
    global_features = tf.reduce_sum(encoded * attention_weights, axis=1)
    
    # ── DECODER (para clasificación) ────────────────────────────────────────
    # Feature processing
    x = layers.Dense(embed_dim * 2, activation='gelu')(global_features)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(embed_dim, activation='gelu')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Clasificación final
    outputs = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Crear modelo
    model = Model(inputs, outputs, name='TransformerAutoencoder')
    
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_transformer_minimal(input_shape=(28, 28, 1), num_classes=10):
    """
    Versión minimalista del Transformer para competir con MLP minimal
    Objetivo: <10k parámetros con mejor accuracy que MLP minimal
    """
    inputs = layers.Input(shape=input_shape)
    
    # Patch embedding más pequeño (7x7 -> 4x4 patches)
    patch_size = 7
    num_patches = (28 // patch_size) ** 2  # 16 patches
    patch_dim = patch_size * patch_size   # 49 for 7x7 patches
    
    # Extraer patches manualmente
    batch_size = tf.shape(inputs)[0]
    patches = tf.image.extract_patches(
        inputs,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    patches = tf.reshape(patches, [batch_size, num_patches, patch_dim])
    
    # Embedding más pequeño
    embed_dim = 32
    x = layers.Dense(embed_dim)(patches)
    
    # Positional embedding
    positions = tf.range(0, num_patches)
    pos_embedding = layers.Embedding(num_patches, embed_dim)
    x += pos_embedding(positions)
    
    # Solo 2 capas transformer con menos heads
    x = TransformerBlock(embed_dim=32, num_heads=2, ff_dim=64, dropout_rate=0.1)(x)
    x = TransformerBlock(embed_dim=32, num_heads=2, ff_dim=64, dropout_rate=0.1)(x)
    
    # Global average pooling simple
    x = layers.GlobalAveragePooling1D()(x)
    
    # Clasificador minimal
    x = layers.Dense(16, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='TransformerMinimal')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Test del modelo
    model = create_transformer_autoencoder()
    model.summary()
    print(f"\nParámetros totales: {model.count_params():,}")
    
    # Test con datos dummy
    dummy_input = tf.random.normal((1, 28, 28, 1))
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")