"""Pequeñas utilidades para visualizar el trade-off sesgo-varianza.

Las curvas son sintéticas (no usan datos del modelo) y sirven para explicar el concepto.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_bias_variance_tradeoff(complexity_points: int = 80, noise: float = 0.05, seed: int = 42):
    """Dibuja curvas de sesgo^2, varianza y error total frente a complejidad.

    Args:
        complexity_points: cantidad de puntos en el eje de complejidad.
        noise: nivel de ruido irreducible que desplaza el error total.
        seed: semilla para reproducibilidad (afecta solo al desplazamiento aleatorio leve).
    """
    rng = np.random.default_rng(seed)
    complexity = np.linspace(0.0, 1.0, complexity_points)

    # Curvas sintéticas y suaves: sesgo decrece, varianza crece con la complejidad.
    bias_sq = np.exp(-3.2 * complexity) + 0.02
    variance = 0.08 * np.exp(2.8 * complexity)

    # Pequeña ondulación para que no sea perfectamente lisa (más realista visualmente).
    wiggle = 0.01 * np.sin(6 * complexity) * (1 + 0.3 * rng.standard_normal(complexity_points))

    total_error = bias_sq + variance + noise + wiggle

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: componentes.
    axes[0].plot(complexity, bias_sq, label="Sesgo^2", color="#1f77b4", linewidth=2)
    axes[0].plot(complexity, variance, label="Varianza", color="#d62728", linewidth=2)
    axes[0].plot(complexity, noise + np.zeros_like(complexity), label="Ruido irreducible", color="#2ca02c", linestyle="--")
    axes[0].set_title("Sesgo y Varianza vs Complejidad")
    axes[0].set_xlabel("Complejidad del modelo")
    axes[0].set_ylabel("Error")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: error total y punto óptimo.
    axes[1].plot(complexity, total_error, label="Error total (b^2 + v + ruido)", color="#9467bd", linewidth=2)
    sweet_spot_idx = int(0.35 * complexity_points)
    axes[1].axvline(complexity[sweet_spot_idx], color="gray", linestyle="--", alpha=0.6, label="Zona óptima")
    axes[1].set_title("Error total y complejidad")
    axes[1].set_xlabel("Complejidad del modelo")
    axes[1].set_ylabel("Error")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig


def plot_bias_variance_single(noise: float = 0.05, seed: int = 42):
    """Versión compacta en un solo panel (útil para notebooks cortos)."""
    rng = np.random.default_rng(seed)
    complexity = np.linspace(0.0, 1.0, 120)
    bias_sq = np.exp(-3.2 * complexity) + 0.02
    variance = 0.08 * np.exp(2.8 * complexity)
    total_error = bias_sq + variance + noise + 0.01 * rng.standard_normal(len(complexity))

    plt.figure(figsize=(7, 4))
    plt.plot(complexity, bias_sq, label="Sesgo^2", color="#1f77b4", linewidth=2)
    plt.plot(complexity, variance, label="Varianza", color="#d62728", linewidth=2)
    plt.plot(complexity, total_error, label="Error total", color="#9467bd", linewidth=2)
    plt.axvline(0.35, color="gray", linestyle="--", alpha=0.6, label="Zona óptima")
    plt.title("Trade-off Sesgo-Varianza")
    plt.xlabel("Complejidad del modelo")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return plt.gcf()
