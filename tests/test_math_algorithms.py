"""
Tests für die mathematischen Algorithmen aus den Jupyter-Notebooks.
=================================================================
Testet alle Funktionen: Gradient Descent, Entropie, Distanzmetriken, etc.
"""

from collections.abc import Callable

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Algorithmen (aus math_algorithms.py extrahiert)
# ═══════════════════════════════════════════════════════════════

def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def normalize(x: np.ndarray) -> np.ndarray:
    x_min, x_max = x.min(), x.max()
    if abs(x_max - x_min) < 1e-10:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def gradient_descent_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float, lr: float = 0.1,
    epochs: int = 100, tol: float = 1e-6
) -> tuple[float, list]:
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
) -> tuple[np.ndarray, list]:
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


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-10, 1)
    return -np.sum(p * np.log2(p))


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log2(p / q))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - cosine_similarity(a, b)

# ═══════════════════════════════════════════════════════════════
# Lineare Algebra
# ═══════════════════════════════════════════════════════════════

class TestSoftmax:
    def test_sums_to_one(self):
        """Softmax-Ausgabe muss sich zu 1 summieren."""
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_preserves_order(self):
        """Größere Eingabewerte ergeben größere Softmax-Werte."""
        x = np.array([1.0, 5.0, 3.0])
        result = softmax(x)
        assert result[1] > result[2] > result[0]

    def test_2d_input(self):
        """Softmax auf 2D-Array (Batch)."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x)
        assert result.shape == (2, 2)
        assert np.allclose(np.sum(result, axis=1), 1.0)

    def test_numerical_stability(self):
        """Softmax mit großen Werten sollte nicht überlaufen."""
        x = np.array([1000.0, 2000.0, 3000.0])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert abs(np.sum(result) - 1.0) < 1e-10


class TestSigmoid:
    def test_range(self):
        """Sigmoid muss Werte in [0, 1] liefern."""
        x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        result = sigmoid(x)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_symmetry(self):
        """sigmoid(0) = 0.5, sigmoid(-x) = 1 - sigmoid(x)."""
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10
        x = np.array([2.0])
        assert abs(sigmoid(-x)[0] - (1 - sigmoid(x)[0])) < 1e-10

    def test_monotonic(self):
        """Sigmoid ist monoton steigend."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sigmoid(x)
        assert np.all(np.diff(result) > 0)


class TestNormalize:
    def test_range(self):
        """Normalisierte Werte müssen in [0, 1] liegen."""
        x = np.array([1.0, 5.0, 10.0])
        result = normalize(x)
        assert abs(result.min()) < 1e-10
        assert abs(result.max() - 1.0) < 1e-10

    def test_constant_input(self):
        """Konstante Eingabe: alle Werte werden 0."""
        x = np.array([5.0, 5.0, 5.0])
        result = normalize(x)
        assert np.allclose(result, 0.0)

    def test_negative_values(self):
        """Funktioniert auch mit negativen Werten."""
        x = np.array([-5.0, 0.0, 5.0])
        result = normalize(x)
        assert abs(result.min()) < 1e-10
        assert abs(result.max() - 1.0) < 1e-10


class TestStandardize:
    def test_zero_mean(self):
        """Standardisierte Werte haben Mittelwert ≈ 0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize(x)
        assert abs(result.mean()) < 1e-10

    def test_unit_variance(self):
        """Standardisierte Werte haben Standardabweichung ≈ 1."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize(x)
        assert abs(result.std() - 1.0) < 1e-8


# ═══════════════════════════════════════════════════════════════
# Gradient Descent
# ═══════════════════════════════════════════════════════════════

class TestGradientDescent1D:
    def test_quadratic(self):
        """Minimum von f(x)=x² bei x=0."""
        def f(x):
            return x**2

        def df(x):
            return 2 * x

        x_min, history = gradient_descent_1d(f, df, x0=5.0, lr=0.1)
        assert abs(x_min) < 1e-4

    def test_history_grows(self):
        """History enthält mindestens Start- und Endwert."""
        def f(x):
            return x**2

        def df(x):
            return 2 * x

        _, history = gradient_descent_1d(f, df, x0=5.0, lr=0.1)
        assert len(history) >= 2

    def test_early_stop(self):
        """Toleranz-basierter Abbruch funktioniert."""
        def f(x):
            return x**2

        def df(x):
            return 2 * x

        x_min, history = gradient_descent_1d(f, df, x0=5.0, lr=0.1, tol=1e-3)
        assert len(history) < 100  # sollte früh stoppen

    def test_shifted_quadratic(self):
        """Minimum von f(x)=(x-3)² bei x=3."""
        def f(x):
            return (x - 3) ** 2

        def df(x):
            return 2 * (x - 3)

        x_min, _ = gradient_descent_1d(f, df, x0=0.0, lr=0.1)
        assert abs(x_min - 3.0) < 1e-4


class TestGradientDescent2D:
    def test_quadratic_2d(self):
        """Minimum von f(x,y)=x²+y² bei (0,0)."""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        x_min, _ = gradient_descent_2d(f, grad_f, np.array([3.0, 4.0]), lr=0.1)
        assert np.linalg.norm(x_min) < 1e-4

    def test_history_contains_copies(self):
        """History-Einträge sind Kopien, keine Referenzen."""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        _, history = gradient_descent_2d(f, grad_f, np.array([3.0, 4.0]))
        # Alle Einträge sollten unterschiedliche Objekte sein
        for i in range(len(history) - 1):
            assert not np.array_equal(history[i], history[i + 1])

    def test_early_stop_2d(self):
        """Toleranz-basierter Abbruch in 2D."""
        def f(x):
            return x[0] ** 2 + x[1] ** 2

        def grad_f(x):
            return np.array([2 * x[0], 2 * x[1]])

        _, history = gradient_descent_2d(f, grad_f, np.array([3.0, 4.0]), lr=0.1, tol=1e-3)
        assert len(history) < 100


# ═══════════════════════════════════════════════════════════════
# Wahrscheinlichkeit & Statistik
# ═══════════════════════════════════════════════════════════════

class TestEntropy:
    def test_uniform(self):
        """Gleichverteilung hat maximale Entropie: log2(n)."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(entropy(p) - 2.0) < 1e-10

    def test_certain(self):
        """Sicheres Ereignis hat Entropie 0."""
        p = np.array([1.0, 0.0, 0.0])
        assert entropy(p) < 1e-8

    def test_non_negative(self):
        """Entropie ist immer ≥ 0."""
        p = np.random.rand(10)
        p = p / p.sum()
        assert entropy(p) >= 0


class TestCrossEntropy:
    def test_perfect_prediction(self):
        """Perfekte Vorhersage: Cross-Entropy ≈ 0."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.999, 0.001])
        assert cross_entropy(y_true, y_pred) < 0.1

    def test_worst_prediction(self):
        """Völlig falsche Vorhersage: hohe Cross-Entropy."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.001, 0.999])
        assert cross_entropy(y_true, y_pred) > 1.0


class TestKLDivergence:
    def test_same_distribution(self):
        """KL-Divergenz identischer Verteilungen ≈ 0."""
        p = np.array([0.5, 0.5])
        assert kl_divergence(p, p) < 1e-10

    def test_non_negative(self):
        """KL-Divergenz ist immer ≥ 0 (Gibbs-Ungleichung)."""
        p = np.array([0.7, 0.3])
        q = np.array([0.3, 0.7])
        assert kl_divergence(p, q) >= 0

    def test_asymmetric(self):
        """KL-Divergenz ist asymmetrisch."""
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        assert abs(kl_divergence(p, q) - kl_divergence(q, p)) > 1e-6


# ═══════════════════════════════════════════════════════════════
# Distanzmetriken
# ═══════════════════════════════════════════════════════════════

class TestEuclideanDistance:
    def test_zero_distance(self):
        """Distanz zum selben Punkt ist 0."""
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) < 1e-10

    def test_known_distance(self):
        """3-4-5 Dreieck: Distanz = 5."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert abs(euclidean_distance(a, b) - 5.0) < 1e-10

    def test_symmetric(self):
        """Distanz ist symmetrisch."""
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        assert abs(euclidean_distance(a, b) - euclidean_distance(b, a)) < 1e-10


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Identische Vektoren: Cosinus = 1."""
        a = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(a, a) - 1.0) < 1e-8

    def test_orthogonal_vectors(self):
        """Orthogonale Vektoren: Cosinus = 0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-10

    def test_opposite_vectors(self):
        """Entgegengesetzte Vektoren: Cosinus = -1."""
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-8

    def test_range(self):
        """Cosinus-Ähnlichkeit liegt in [-1, 1]."""
        a = np.random.randn(10)
        b = np.random.randn(10)
        sim = cosine_similarity(a, b)
        assert -1.0 - 1e-10 <= sim <= 1.0 + 1e-10


class TestCosineDistance:
    def test_identical(self):
        """Identische Vektoren: Distanz = 0."""
        a = np.array([1.0, 2.0])
        assert cosine_distance(a, a) < 1e-8

    def test_orthogonal(self):
        """Orthogonale Vektoren: Distanz = 1."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_distance(a, b) - 1.0) < 1e-10
