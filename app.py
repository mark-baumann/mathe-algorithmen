"""
Streamlit-App: Mathe-Algorithmen für Machine Learning
======================================================
Gradient Descent animieren, Entropie berechnen, Distanzmetriken vergleichen.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from math_algorithms import (
    gradient_descent_1d, gradient_descent_2d,
    entropy, cross_entropy, kl_divergence,
    euclidean_distance, cosine_similarity, cosine_distance,
    softmax, sigmoid, normalize, standardize,
)

st.set_page_config(page_title="Mathe-Algorithmen", page_icon="📐", layout="wide")
st.title("📐 Mathe-Algorithmen für Machine Learning")
st.markdown("Gradient Descent, Entropie, Distanzmetriken — die mathematischen Grundlagen der KI")

page = st.sidebar.radio(
    "Bereich wählen",
    ["Gradient Descent", "Entropie & Information", "Distanzmetriken", "Aktivierungsfunktionen"]
)

# ═══════════════════════════════════════════════════════════════════════════
# GRADIENT DESCENT
# ═══════════════════════════════════════════════════════════════════════════
if page == "Gradient Descent":
    st.header("📉 Gradient Descent — Interaktiv")

    st.markdown("""
    **Gradient Descent** ist der fundamentale Optimierungsalgorithmus für Machine Learning.
    
    Die Idee: Folge dem negativen Gradienten, um das Minimum einer Funktion zu finden.
    
    **Update-Regel:** `x_new = x_old - learning_rate * gradient`
    """)

    tab1, tab2 = st.tabs(["1D (eine Variable)", "2D (zwei Variablen)"])

    with tab1:
        st.subheader("1D-Gradient-Descent: f(x) = x²")

        col1, col2, col3 = st.columns(3)
        with col1:
            x0 = st.slider("Startwert x₀", -10.0, 10.0, 5.0, 0.5)
        with col2:
            lr = st.selectbox("Learning-Rate", [0.01, 0.05, 0.1, 0.2, 0.5, 0.9], index=2, key="lr1d")
        with col3:
            epochs = st.slider("Epochen", 5, 50, 20, key="ep1d")

        if st.button("Gradient Descent starten", key="gd1d"):
            def f(x):
                return x**2
            def df(x):
                return 2*x

            x_min, history = gradient_descent_1d(f, df, x0, lr=lr, epochs=epochs)

            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Funktionsplot mit Pfad
            x_range = np.linspace(-max(abs(x0)*1.5, 10), max(abs(x0)*1.5, 10), 500)
            ax1.plot(x_range, f(x_range), "b-", linewidth=2, label="f(x) = x²")
            ax1.scatter(history, [f(h) for h in history], c=range(len(history)),
                       cmap="Reds", s=50, zorder=5, edgecolors="white")
            ax1.scatter([x_min], [f(x_min)], color="green", s=200, marker="*",
                       zorder=6, label=f"Minimum: x={x_min:.4f}")
            ax1.set_xlabel("x")
            ax1.set_ylabel("f(x)")
            ax1.set_title("Gradient-Descent-Pfad auf f(x) = x²")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Konvergenz
            ax2.plot(history, "r-o", linewidth=2, markersize=4)
            ax2.axhline(y=0, color="green", linestyle="--", label="Optimum (x=0)")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("x-Wert")
            ax2.set_title("Konvergenz von x gegen 0")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)

            st.success(f"Minimum gefunden bei x = {x_min:.6f} (erwartet: 0)")

    with tab2:
        st.subheader("2D-Gradient-Descent: f(x,y) = x² + y²")

        col1, col2 = st.columns(2)
        with col1:
            x0_2d = st.slider("Startwert x₀", -10.0, 10.0, 3.0, 0.5, key="x0_2d")
        with col2:
            y0_2d = st.slider("Startwert y₀", -10.0, 10.0, 4.0, 0.5, key="y0_2d")

        lr_2d = st.selectbox("Learning-Rate", [0.01, 0.05, 0.1, 0.2, 0.5], index=2, key="lr2d")

        if st.button("Gradient Descent starten", key="gd2d"):
            def f2d(x):
                return x[0]**2 + x[1]**2
            def grad_f2d(x):
                return np.array([2*x[0], 2*x[1]])

            x0 = np.array([x0_2d, y0_2d])
            x_min, history = gradient_descent_2d(f2d, grad_f2d, x0, lr=lr_2d, epochs=50)
            history = np.array(history)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Contour-Plot
            r = max(abs(x0_2d), abs(y0_2d)) * 1.5
            x_vals = np.linspace(-r, r, 100)
            y_vals = np.linspace(-r, r, 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = X**2 + Y**2

            ax1.contour(X, Y, Z, levels=20, cmap="Blues", alpha=0.6)
            ax1.plot(history[:, 0], history[:, 1], "r-o", linewidth=2, markersize=4)
            ax1.scatter([0], [0], color="green", s=200, marker="*", zorder=6, label="Minimum (0,0)")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_title("Gradient-Descent-Pfad (Contour)")
            ax1.legend()
            ax1.set_aspect("equal")

            # Konvergenz
            distances = np.sqrt(history[:, 0]**2 + history[:, 1]**2)
            ax2.plot(distances, "r-o", linewidth=2, markersize=4)
            ax2.axhline(y=0, color="green", linestyle="--", label="Optimum")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Distanz zum Optimum")
            ax2.set_title("Konvergenz: Distanz zu (0,0)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)

            st.success(f"Minimum gefunden bei ({x_min[0]:.6f}, {x_min[1]:.6f}) (erwartet: (0, 0))")

# ═══════════════════════════════════════════════════════════════════════════
# ENTROPIE & INFORMATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Entropie & Information":
    st.header("📊 Entropie & Informationstheorie")

    st.markdown("""
    **Shannon-Entropie** misst die Unsicherheit einer Wahrscheinlichkeitsverteilung.
    
    H = -Σ pᵢ · log₂(pᵢ)
    
    - **Hohe Entropie** = hohe Unsicherheit (z.B. fairer Münzwurf: 1 Bit)
    - **Niedrige Entropie** = hohe Sicherheit (z.B. gezinkte Münze: nahe 0 Bit)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Entropie berechnen")
        st.markdown("Gib Wahrscheinlichkeiten ein (Summe = 1):")

        p1 = st.slider("p₁", 0.0, 1.0, 0.5, 0.01)
        p2 = 1.0 - p1

        probs = np.array([p1, p2])
        ent = entropy(probs)

        st.metric("Entropie", f"{ent:.4f} Bit")

        # Visualisierung
        p_range = np.linspace(0.01, 0.99, 100)
        entropies = [entropy(np.array([p, 1-p])) for p in p_range]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(p_range, entropies, "b-", linewidth=2)
        ax.scatter([p1], [ent], color="red", s=100, zorder=5)
        ax.set_xlabel("p₁ (Wahrscheinlichkeit Klasse 1)")
        ax.set_ylabel("Entropie (Bit)")
        ax.set_title("Entropie einer Binärverteilung")
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Maximale Entropie")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        st.subheader("KL-Divergenz")
        st.markdown("""
        **Kullback-Leibler-Divergenz** misst, wie sehr sich zwei Verteilungen unterscheiden.
        
        D_KL(P||Q) = Σ pᵢ · log₂(pᵢ/qᵢ)
        
        - D_KL = 0 wenn P = Q
        - D_KL > 0 sonst (asymmetrisch!)
        """)

        st.markdown("**Verteilung P (wahre Verteilung):**")
        p_a = st.slider("P(Klasse 1)", 0.01, 0.99, 0.7, 0.01, key="pa")

        st.markdown("**Verteilung Q (Modell-Verteilung):**")
        q_a = st.slider("Q(Klasse 1)", 0.01, 0.99, 0.3, 0.01, key="qa")

        P = np.array([p_a, 1-p_a])
        Q = np.array([q_a, 1-q_a])
        kl = kl_divergence(P, Q)

        st.metric("D_KL(P||Q)", f"{kl:.4f} Bit")

        st.markdown("""
        **Interpretation:**
        - D_KL = 0: Perfekte Übereinstimmung
        - D_KL klein: Ähnliche Verteilungen
        - D_KL groß: Sehr unterschiedliche Verteilungen
        """)

# ═══════════════════════════════════════════════════════════════════════════
# DISTANZMETRIKEN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Distanzmetriken":
    st.header("📏 Distanzmetriken vergleichen")

    st.markdown("""
    Distanzmetriken messen, wie "ähnlich" oder "verschieden" zwei Vektoren sind.
    Die Wahl der Metrik beeinflusst ML-Algorithmen wie k-NN, Clustering und Embedding-Vergleiche.
    """)

    st.subheader("Interaktiver Vergleich")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vektor A**")
        a1 = st.slider("A[0]", -5.0, 5.0, 1.0, 0.5, key="a1")
        a2 = st.slider("A[1]", -5.0, 5.0, 0.0, 0.5, key="a2")

    with col2:
        st.markdown("**Vektor B**")
        b1 = st.slider("B[0]", -5.0, 5.0, 0.0, 0.5, key="b1")
        b2 = st.slider("B[1]", -5.0, 5.0, 1.0, 0.5, key="b2")

    a = np.array([a1, a2])
    b = np.array([b1, b2])

    eucl = euclidean_distance(a, b)
    cos_sim = cosine_similarity(a, b)
    cos_dist = cosine_distance(a, b)
    manhattan = np.sum(np.abs(a - b))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Euklidisch", f"{eucl:.3f}")
    with col2:
        st.metric("Manhattan", f"{manhattan:.3f}")
    with col3:
        st.metric("Cosinus-Ähnlichkeit", f"{cos_sim:.3f}")
    with col4:
        st.metric("Cosinus-Distanz", f"{cos_dist:.3f}")

    # Visualisierung
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1,
              color="#FF6B6B", width=0.02, label=f"A ({a[0]}, {a[1]})")
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1,
              color="#4ECDC4", width=0.02, label=f"B ({b[0]}, {b[1]})")

    # Euklidische Distanz als gestrichelte Linie
    ax.plot([a[0], b[0]], [a[1], b[1]], "gray", linestyle="--", alpha=0.5,
            label=f"Euklidisch: {eucl:.2f}")

    r = max(abs(a1), abs(a2), abs(b1), abs(b2)) * 1.3
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Vektoren und ihre Distanzen")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.5)

    st.pyplot(fig)

    st.subheader("Metriken im Überblick")
    st.markdown("""
    | Metrik | Formel | Anwendung |
    |--------|--------|-----------|
    | **Euklidisch (L2)** | √Σ(aᵢ - bᵢ)² | k-NN, Clustering, allgemein |
    | **Manhattan (L1)** | Σ\|aᵢ - bᵢ\| | Hochdimensionale Daten, robust gegen Ausreißer |
    | **Cosinus-Ähnlichkeit** | a·b / (\|\|a\|\|·\|\|b\|\|) | Text-Embeddings, Empfehlungssysteme |
    | **Cosinus-Distanz** | 1 - Cosinus-Ähnlichkeit | Gleiche Anwendung wie Cosinus-Ähnlichkeit |
    """)

# ═══════════════════════════════════════════════════════════════════════════
# AKTIVIERUNGSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Aktivierungsfunktionen":
    st.header("🔧 Aktivierungsfunktionen")

    st.markdown("""
    Aktivierungsfunktionen bringen Nichtlinearität in neuronale Netze.
    Ohne sie wäre jedes NN nur eine lineare Transformation!
    """)

    func = st.selectbox("Funktion", ["Sigmoid", "Softmax", "ReLU (Vergleich)"])

    x = np.linspace(-5, 5, 200)

    if func == "Sigmoid":
        y = sigmoid(x)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, linewidth=2, color="#FF6B6B")
        ax.set_title("Sigmoid: f(x) = 1/(1+e^(-x))")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        st.pyplot(fig)

        st.markdown("""
        **Eigenschaften:**
        - Ausgabe: (0, 1) — gut für Wahrscheinlichkeiten
        - Problem: "Vanishing Gradient" bei großen |x|
        - Früher Standard, heute meist durch ReLU ersetzt
        """)

    elif func == "Softmax":
        logits = np.array([2.0, 1.0, 0.1, -0.5, -1.0])
        probs = softmax(logits)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.bar(range(5), logits, color="#4ECDC4", edgecolor="white")
        ax1.set_title("Logits (Rohwerte)")
        ax1.set_xlabel("Klasse")
        ax1.set_ylabel("Wert")

        ax2.bar(range(5), probs, color="#FF6B6B", edgecolor="white")
        ax2.set_title("Softmax (Wahrscheinlichkeiten)")
        ax2.set_xlabel("Klasse")
        ax2.set_ylabel("Wahrscheinlichkeit")
        ax2.set_ylim(0, 1)

        st.pyplot(fig)

        st.markdown("""
        **Eigenschaften:**
        - Wandelt Logits in Wahrscheinlichkeiten um (Summe = 1)
        - Numerisch stabil durch Subtraktion des Maximums
        - Standard für Multi-Klassen-Klassifikation
        """)

    else:
        y_relu = np.maximum(0, x)
        y_sigmoid = sigmoid(x)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(x, y_relu, linewidth=2, color="#FF6B6B", label="ReLU")
        ax1.plot(x, y_sigmoid, linewidth=2, color="#4ECDC4", label="Sigmoid")
        ax1.set_title("ReLU vs. Sigmoid")
        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gradienten
        relu_grad = (x > 0).astype(float)
        sigmoid_grad = y_sigmoid * (1 - y_sigmoid)

        ax2.plot(x, relu_grad, linewidth=2, color="#FF6B6B", label="ReLU-Gradient")
        ax2.plot(x, sigmoid_grad, linewidth=2, color="#4ECDC4", label="Sigmoid-Gradient")
        ax2.set_title("Gradienten im Vergleich")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f'(x)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

        st.markdown("""
        **ReLU-Vorteile:**
        - Kein Vanishing Gradient für x > 0
        - Schnellere Berechnung (kein exp)
        - Führt zu sparsen Aktivierungen (viele Nullen)
        
        **ReLU-Nachteil:**
        - "Dying ReLU": Neuronen mit x ≤ 0 lernen nie wieder
        """)

st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Mathe-Algorithmen für ML**")
st.sidebar.markdown("[GitHub Repository](https://github.com/mark-baumann/mathe-algorithmen)")
