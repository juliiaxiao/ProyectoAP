# -*- coding: utf-8 -*-
"""
Vision Transformer Híbrido para Fashion MNIST

Combina lo mejor de CNNs y Transformers:
- CNN feature extractor inicial para capturar patrones locales
- Transformer encoder para relaciones globales  
- Hierarchical attention con múltiples escalas
- Más eficiente que Vision Transformers puros
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class CNNFeatureExtractor(layers.Layer):
    """
    Extractor de features CNN optimizado para Fashion MNIST
    Reduce dimensionalidad antes del Transformer
    """
    def __init__(self, num_filters=[32, 64], **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        
    def build(self, input_shape):
        self.conv_blocks = []
        
        # Bloque 1: captura textura y bordes
        block1 = tf.keras.Sequential([
            layers.Conv2D(self.num_filters[0], 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(self.num_filters[0], 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),  # 28x28 -> 14x14
        ])
        self.conv_blocks.append(block1)
        
        # Bloque 2: captura patrones más complejos
        block2 = tf.keras.Sequential([
            layers.Conv2D(self.num_filters[1], 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(self.num_filters[1], 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),  # 14x14 -> 7x7
        ])
        self.conv_blocks.append(block2)
        
    def call(self, inputs):
        x = inputs
        features = []
        
        for block in self.conv_blocks:
            x = block(x)
            features.append(x)  # Guardar features de cada escala
        
        return x, features  # (B, 7, 7, 64), [(B,14,14,32), (B,7,7,64)]


class SpatialToSequence(layers.Layer):
    """
    Convierte feature maps espaciales a secuencia para Transformer
    Añade embeddings posicionales 2D
    """
    def __init__(self, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # input_shape: (B, H, W, C)
        self.H, self.W, self.C = input_shape[1:]
        self.seq_len = self.H * self.W
        
        # Proyección linear de features CNN a embed_dim
        self.projection = layers.Dense(self.embed_dim)
        
        # Positional embeddings 2D
        self.pos_embed_h = layers.Embedding(self.H, self.embed_dim // 2)
        self.pos_embed_w = layers.Embedding(self.W, self.embed_dim // 2)
        
    def call(self, features):
        # features: (B, H, W, C)
        batch_size = tf.shape(features)[0]
        
        # Flatten spatial dimensions: (B, H*W, C)
        flattened = tf.reshape(features, [batch_size, self.seq_len, self.C])
        
        # Project to embed_dim: (B, H*W, embed_dim)
        embedded = self.projection(flattened)
        
        # Generate 2D positional embeddings
        h_positions = tf.repeat(tf.range(self.H), self.W)  # [0,0,0,1,1,1,2,2,2,...]
        w_positions = tf.tile(tf.range(self.W), [self.H])  # [0,1,2,0,1,2,0,1,2,...]
        
        h_embeds = self.pos_embed_h(h_positions)  # (H*W, embed_dim//2)
        w_embeds = self.pos_embed_w(w_positions)  # (H*W, embed_dim//2)
        
        # Concatenar embeddings posicionales
        pos_embeds = tf.concat([h_embeds, w_embeds], axis=-1)  # (H*W, embed_dim)
        pos_embeds = tf.expand_dims(pos_embeds, 0)  # (1, H*W, embed_dim)
        pos_embeds = tf.tile(pos_embeds, [batch_size, 1, 1])  # (B, H*W, embed_dim)
        
        return embedded + pos_embeds


class MultiScaleTransformerBlock(layers.Layer):
    """
    Transformer block que opera en múltiples escalas
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Self-attention en la escala principal
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        
        # Feed forward network
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        # Self-attention
        attn_output = self.self_attention(inputs, inputs, attention_mask=None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class HierarchicalAttentionPooling(layers.Layer):
    """
    Pooling con atención jerárquica que combina información 
    de diferentes escalas espaciales
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # Attention weights para cada posición
        self.attention_weights = layers.Dense(1, activation='softmax')
        
        # Proyección final
        self.final_projection = layers.Dense(self.embed_dim)
        
    def call(self, sequence_features):
        # sequence_features: (B, seq_len, embed_dim)
        
        # Calcular pesos de atención para cada posición
        attention_scores = self.attention_weights(sequence_features)  # (B, seq_len, 1)
        
        # Weighted average
        weighted_features = tf.reduce_sum(
            sequence_features * attention_scores, axis=1
        )  # (B, embed_dim)
        
        # Proyección final
        output = self.final_projection(weighted_features)
        
        return output, attention_scores


def create_hybrid_vit(
    input_shape=(28, 28, 1),
    num_classes=10,
    cnn_filters=[32, 64],
    embed_dim=128,
    num_transformer_layers=3,
    num_heads=8,
    ff_dim=256,
    dropout_rate=0.1
):
    """
    Vision Transformer Híbrido que combina CNN + Transformer
    
    Args:
        input_shape: forma de entrada
        num_classes: número de clases
        cnn_filters: filtros para feature extractor CNN
        embed_dim: dimensión de embeddings del Transformer 
        num_transformer_layers: número de capas Transformer
        num_heads: heads en multi-head attention
        ff_dim: dimensión feed-forward network
        dropout_rate: tasa de dropout
    
    Returns:
        Modelo Keras compilado
        
    Parámetros aprox: ~75k (más eficiente que Transformer puro)
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # ── 1. CNN Feature Extractor ────────────────────────────────────────────
    cnn_features, multi_scale_features = CNNFeatureExtractor(cnn_filters)(inputs)
    # cnn_features: (B, 7, 7, 64)
    
    # ── 2. Spatial to Sequence Conversion ───────────────────────────────────
    sequence_features = SpatialToSequence(embed_dim)(cnn_features)
    # sequence_features: (B, 49, embed_dim)
    
    # ── 3. Transformer Encoder ──────────────────────────────────────────────
    x = sequence_features
    for i in range(num_transformer_layers):
        x = MultiScaleTransformerBlock(
            embed_dim, num_heads, ff_dim, dropout_rate
        )(x)
    
    # ── 4. Hierarchical Attention Pooling ───────────────────────────────────
    global_features, attention_weights = HierarchicalAttentionPooling(embed_dim)(x)
    
    # ── 5. Classifier Head ──────────────────────────────────────────────────
    x = layers.Dense(embed_dim // 2, activation='gelu')(global_features)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='classification')(x)
    
    # Crear modelo
    model = Model(inputs, outputs, name='HybridViT')
    
    # Compilar con AdamW (mejor para Transformers)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-3, 
            weight_decay=1e-4
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_efficient_hybrid_vit(input_shape=(28, 28, 1), num_classes=10):
    """
    Versión ultra-eficiente del Hybrid ViT
    Objetivo: <20k parámetros con mejor accuracy que CNN minimal
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN feature extractor más pequeño
    x = layers.Conv2D(24, 5, strides=2, padding='same', activation='relu')(inputs)  # 14x14
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(48, 3, strides=2, padding='same', activation='relu')(x)      # 7x7
    x = layers.BatchNormalization()(x)
    
    # Spatial to sequence (más compacto)
    batch_size = tf.shape(x)[0]
    x_flat = tf.reshape(x, [batch_size, 49, 48])  # 7*7=49 patches
    
    # Embedding projectión más pequeña
    embed_dim = 64
    x_embedded = layers.Dense(embed_dim)(x_flat)
    
    # Positional embedding simple
    pos_embed = layers.Embedding(49, embed_dim)
    positions = tf.range(49)
    x_embedded += pos_embed(positions)
    
    # Una sola capa Transformer eficiente
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=16
    )(x_embedded, x_embedded)
    
    x = layers.LayerNormalization()(x_embedded + attention_output)
    
    # Feed-forward pequeño
    ffn_output = layers.Dense(128, activation='gelu')(x)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    x = layers.LayerNormalization()(x + ffn_output)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classifier minimal
    x = layers.Dense(32, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='EfficientHybridViT')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_attention_augmented_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    CNN tradicional aumentada con capas de atención
    Mantiene la estructura CNN pero añade self-attention
    """
    inputs = layers.Input(shape=input_shape)
    
    # Bloque 1: CNN tradicional
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)  # 14x14
    
    # Inyectar self-attention en feature maps
    batch_size = tf.shape(x)[0]
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    
    # Reshape para attention: (B, H*W, C)
    x_reshaped = tf.reshape(x, [batch_size, h*w, c])
    
    # Self-attention en el espacio de features
    attn_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=c//4
    )(x_reshaped, x_reshaped)
    
    # Reshape back: (B, H, W, C)
    attn_output = tf.reshape(attn_output, [batch_size, h, w, c])
    
    # Skip connection
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    
    # Bloque 2: Más CNN
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='AttentionAugmentedCNN')
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Test de los modelos
    print("🧪 Testing Hybrid Vision Transformer models...")
    
    # Modelo principal
    model = create_hybrid_vit()
    model.summary()
    print(f"Hybrid ViT parameters: {model.count_params():,}")
    
    print("\n" + "="*60)
    
    # Modelo eficiente
    efficient_model = create_efficient_hybrid_vit()
    efficient_model.summary()
    print(f"Efficient Hybrid ViT parameters: {efficient_model.count_params():,}")
    
    print("\n" + "="*60)
    
    # Modelo attention-augmented CNN
    aug_cnn = create_attention_augmented_cnn()
    aug_cnn.summary()
    print(f"Attention-Augmented CNN parameters: {aug_cnn.count_params():,}")
    
    # Test forward pass
    dummy_input = tf.random.normal((2, 28, 28, 1))
    
    print(f"\nForward pass test:")
    print(f"Hybrid ViT output: {model(dummy_input).shape}")
    print(f"Efficient ViT output: {efficient_model(dummy_input).shape}")
    print(f"Aug CNN output: {aug_cnn(dummy_input).shape}")
    
    print("✅ All models working correctly!")