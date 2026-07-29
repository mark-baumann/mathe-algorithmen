"""
Mathematische Algorithmen für Machine Learning
==============================================
Gradient Descent, Matrix-Operationen, Wahrscheinlichkeit —
die mathematischen Grundlagen für KI.
"""

import numpy as np
from typing import Callable, Tuple


# ═══════════════════════════════════════════════════════════════
# Lineare Algebra
# ═══════════════════════════════════════════════════════════════

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax mit numerischer Stabilität."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid: 1/(1+e^(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def normalize(x: np.ndarray) -> np.ndarray:
    """Min-Max-Normalisierung auf [0, 1]."""
    x_min, x_max = x.min(), x.max()
    if abs(x_max - x_min) < 1e-10:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def standardize(x: np.ndarray) -> np.ndarray:
    """Z-Score-Standardisierung: (x - μ) / σ"""
    return (x - x.mean()) / (x.std() + 1e-8)


# ═══════════════════════════════════════════════════════════════
# Gradient Descent
# ═══════════════════════════════════════════════════════════════

def gradient_descent_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float, lr: float = 0.1,
    epochs: int = 100, tol: float = 1e-6
) -> Tuple[float, list]:
    """
    Gradient Descent für 1D-Funktionen.
    Findet das Minimum von f(x).

    Returns:
        (x_min, history) — Minimum und Verlauf der x-Werte
    """
    x = x0
    history = [x]
    for _ in range(epochs):
        grad = df(x)
        x_new = x - lr * grad
        history.append(x_new)
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, history


def gradient_descent_2d(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray, lr: float = 0.1,
    epochs: int = 100, tol: float = 1e-6
) -> Tuple[np.ndarray, list]:
    """
    Gradient Descent für mehrdimensionale Funktionen.

    Args:
        f: Zielfunktion f(x) -> float
        grad_f: Gradient ∇f(x) -> np.ndarray
        x0: Startpunkt
    """
    x = x0.copy()
    history = [x.copy()]
    for _ in range(epochs):
        g = grad_f(x)
        x_new = x - lr * g
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x, history


# ═══════════════════════════════════════════════════════════════
# Wahrscheinlichkeit & Statistik
# ═══════════════════════════════════════════════════════════════

def entropy(p: np.ndarray) -> float:
    """Shannon-Entropie: H = -Σ p_i * log(p_i)"""
    p = np.clip(p, 1e-10, 1)
    return -np.sum(p * np.log2(p))


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross-Entropy Loss."""
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler-Divergenz: D_KL(P||Q) in Bits (log2)."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log2(p / q))


# ═══════════════════════════════════════════════════════════════
# Distanzmetriken
# ═══════════════════════════════════════════════════════════════

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euklidische Distanz: ||a - b||₂"""
    return np.sqrt(np.sum((a - b) ** 2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosinus-Ähnlichkeit: a·b / (||a||·||b||)"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosinus-Distanz: 1 - cosine_similarity"""
    return 1 - cosine_similarity(a, b)


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Mathe-Algorithmen für ML")
    print("=" * 50)

    # Softmax
    logits = np.array([2.0, 1.0, 0.1])
    print(f"\nSoftmax({logits}): {softmax(logits)}")

    # Gradient Descent: f(x) = x², f'(x) = 2x
    f = lambda x: x**2
    df = lambda x: 2 * x
    x_min, hist = gradient_descent_1d(f, df, x0=5.0, lr=0.1)
    print(f"\nGD: min von x² bei x={x_min:.6f} (erwartet: 0)")

    # 2D: f(x,y) = x² + y²
    f2d = lambda x: x[0]**2 + x[1]**2
    grad_f2d = lambda x: np.array([2*x[0], 2*x[1]])
    x_min_2d, _ = gradient_descent_2d(f2d, grad_f2d, np.array([3.0, 4.0]))
    print(f"GD 2D: min von x²+y² bei {x_min_2d} (erwartet: [0,0])")

    # Entropie
    p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
    p_certain = np.array([0.99, 0.01])
    print(f"\nEntropie (uniform): {entropy(p_uniform):.4f} bit")
    print(f"Entropie (sicher):  {entropy(p_certain):.4f} bit")

    # Distanzen
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    print(f"\nEuclidean(a,b): {euclidean_distance(a, b):.4f}")
    print(f"Cosine(a,b):    {cosine_similarity(a, b):.4f}")
    print(f"Cosine(a,c):    {cosine_similarity(a, c):.4f} (identisch)")
