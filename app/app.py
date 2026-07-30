"""
Streamlit-App: Mathematische Algorithmen für Machine Learning
=============================================================
Gradient Descent animieren, Entropie berechnen, Distanzmetriken vergleichen.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from math_algorithms import (
    cosine_distance,
    cosine_similarity,
    cross_entropy,
    entropy,
    euclidean_distance,
    gradient_descent_1d,
    gradient_descent_2d,
    kl_divergence,
    sigmoid,
    softmax,
)

st.set_page_config(page_title="Mathe-Algorithmen für ML", layout="wide")
st.title("📐 Mathematische Algorithmen für Machine Learning")
st.markdown("### Gradient Descent, Entropie & Distanzmetriken — interaktiv")

tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Gradient Descent",
    "🎲 Entropie & Information",
    "📏 Distanzmetriken",
    "🧮 Aktivierungsfunktionen",
])

# ═══════════════════════════════════════════════════════════════
# Tab 1: Gradient Descent
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("📉 Gradient Descent — interaktiv")

    st.markdown(r"""
    **Gradient Descent** ist der fundamentale Optimierungsalgorithmus im Machine Learning.
    Er findet das Minimum einer Funktion, indem er iterativ dem negativen Gradienten folgt.

    **Update-Regel:** $x_{t+1} = x_t - \eta \cdot \nabla f(x_t)$

    - $\eta$ (Learning Rate): Schrittweite
    - $\nabla f(x_t)$: Gradient (Richtung des steilsten Anstiegs)
    """)

    st.subheader("1D Gradient Descent: $f(x) = x^2$")

    col_gd1, col_gd2 = st.columns([1, 2])

    with col_gd1:
        x0 = st.slider("Startpunkt $x_0$", -10.0, 10.0, 5.0, 0.5)
        lr = st.select_slider(
            "Learning Rate η",
            options=[0.01, 0.05, 0.1, 0.2, 0.5, 0.9],
            value=0.1,
        )
        n_steps = st.slider("Anzahl Schritte", 5, 50, 20)

    with col_gd2:
        # Führe GD aus
        def f(x):
            return x**2
        def df(x):
            return 2 * x

        x_min, history = gradient_descent_1d(f, df, x0=x0, lr=lr, epochs=n_steps)

        # Plot
        fig_gd, ax_gd = plt.subplots(figsize=(10, 6))
        x_range = np.linspace(-max(abs(x0)*1.5, 10), max(abs(x0)*1.5, 10), 500)
        ax_gd.plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x) = x²')
        ax_gd.scatter(history, [f(x) for x in history], c=range(len(history)),
                      cmap='Reds', s=50, zorder=5, edgecolors='black')
        ax_gd.plot(history, [f(x) for x in history], 'r--', alpha=0.5, linewidth=1)
        ax_gd.scatter([x_min], [f(x_min)], c='green', s=200, zorder=6,
                      edgecolors='black', marker='*', label=f'Minimum: x={x_min:.4f}')
        ax_gd.set_xlabel('x')
        ax_gd.set_ylabel('f(x)')
        ax_gd.set_title(f'Gradient Descent: {len(history)} Schritte → Minimum bei x={x_min:.4f}',
                        fontweight='bold')
        ax_gd.legend()
        ax_gd.grid(True, alpha=0.3)
        st.pyplot(fig_gd)

        st.metric("Gefundenes Minimum", f"x = {x_min:.6f}", "Ziel: 0.0")

    st.subheader("2D Gradient Descent: $f(x, y) = x^2 + y^2$")

    col_gd2d1, col_gd2d2 = st.columns([1, 2])

    with col_gd2d1:
        x0_2d = st.slider("Startpunkt x₀", -10.0, 10.0, 3.0, 0.5, key="x0_2d")
        y0_2d = st.slider("Startpunkt y₀", -10.0, 10.0, 4.0, 0.5, key="y0_2d")
        lr_2d = st.select_slider(
            "Learning Rate",
            options=[0.01, 0.05, 0.1, 0.2, 0.5],
            value=0.1,
            key="lr_2d",
        )
        n_steps_2d = st.slider("Schritte", 5, 50, 15, key="steps_2d")

    with col_gd2d2:
        def f2d(x):
            return x[0]**2 + x[1]**2
        def grad_f2d(x):
            return np.array([2*x[0], 2*x[1]])

        x_min_2d, history_2d = gradient_descent_2d(
            f2d, grad_f2d, np.array([x0_2d, y0_2d]), lr=lr_2d, epochs=n_steps_2d
        )

        # Contour-Plot
        fig_2d, ax_2d = plt.subplots(figsize=(8, 7))
        max_abs = max(abs(x0_2d), abs(y0_2d)) * 1.5
        x_range_2d = np.linspace(-max_abs, max_abs, 200)
        y_range_2d = np.linspace(-max_abs, max_abs, 200)
        X, Y = np.meshgrid(x_range_2d, y_range_2d)
        Z = X**2 + Y**2

        contour = ax_2d.contour(X, Y, Z, levels=20, cmap='Blues', alpha=0.6)
        ax_2d.clabel(contour, inline=True, fontsize=8)

        hist_arr = np.array(history_2d)
        ax_2d.plot(
            hist_arr[:, 0], hist_arr[:, 1],
            'r-o', markersize=6, linewidth=2, label='GD-Pfad',
        )
        ax_2d.scatter([x0_2d], [y0_2d], c='orange', s=150, zorder=5, edgecolors='black',
                      marker='s', label=f'Start ({x0_2d}, {y0_2d})')
        ax_2d.scatter(
            [x_min_2d[0]], [x_min_2d[1]],
            c='green', s=200, zorder=5, edgecolors='black', marker='*',
            label=f'Min ({x_min_2d[0]:.3f}, {x_min_2d[1]:.3f})',
        )
        ax_2d.set_xlabel('x')
        ax_2d.set_ylabel('y')
        ax_2d.set_title('2D Gradient Descent auf f(x,y) = x² + y²', fontweight='bold')
        ax_2d.legend()
        ax_2d.grid(True, alpha=0.3)
        st.pyplot(fig_2d)

    st.subheader("📖 Learning Rate: Zu klein vs. zu groß")
    st.markdown("""
    | Learning Rate | Effekt |
    |--------------|--------|
    | Zu klein (0.001) | 🐌 Sehr langsame Konvergenz, viele Schritte nötig |
    | Optimal (0.1) | ✅ Schnelle, stabile Konvergenz |
    | Zu groß (0.9) | 🎢 Oszillation, springt über Minimum hinweg |
    | Viel zu groß (2.0) | 💥 Divergenz, entfernt sich vom Minimum |
    """)

# ═══════════════════════════════════════════════════════════════
# Tab 2: Entropie & Information
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("🎲 Entropie & Informationstheorie")

    st.markdown(r"""
    **Entropie** misst die Unsicherheit einer Wahrscheinlichkeitsverteilung.

    **Shannon-Entropie:** $H(P) = -\sum_i p_i \cdot \log_2(p_i)$

    - $H = 0$: völlige Sicherheit (eine Klasse hat $p=1$)
    - $H = \log_2(n)$: maximale Unsicherheit (Gleichverteilung)
    """)

    st.subheader("Entropie-Rechner")

    col_ent1, col_ent2 = st.columns(2)

    with col_ent1:
        st.markdown("**Eigene Verteilung eingeben:**")
        n_classes = st.slider("Anzahl Klassen", 2, 10, 4, key="n_classes_ent")
        probs = []
        remaining = 1.0
        for i in range(n_classes - 1):
            p = st.slider(
                f"p_{i+1}",
                0.0, remaining,
                remaining / (n_classes - i),
                0.01,
                key=f"p_{i}",
            )
            probs.append(p)
            remaining -= p
        probs.append(remaining)
        st.caption(f"p_{n_classes} = {remaining:.3f} (Rest)")

        p_array = np.array(probs)
        H = entropy(p_array)
        H_max = np.log2(n_classes)

        st.metric("Entropie H", f"{H:.4f} bit")
        st.metric("Maximale Entropie", f"{H_max:.4f} bit")
        st.metric("Relative Entropie", f"{H/H_max:.1%}")

    with col_ent2:
        fig_ent, ax_ent = plt.subplots(figsize=(6, 5))
        ax_ent.bar(
            range(1, n_classes + 1), p_array,
            color=plt.cm.viridis(np.linspace(0.2, 0.8, n_classes)),
        )
        ax_ent.set_xlabel('Klasse')
        ax_ent.set_ylabel('Wahrscheinlichkeit')
        ax_ent.set_title(f'Verteilung (Entropie = {H:.3f} bit)', fontweight='bold')
        ax_ent.set_ylim(0, 1)
        st.pyplot(fig_ent)

    st.subheader("Entropie-Vergleich: Sicher vs. Unsicher")
    col_comp1, col_comp2, col_comp3 = st.columns(3)

    with col_comp1:
        p_certain = np.array([0.99, 0.01])
        H_certain = entropy(p_certain)
        st.metric("Sichere Verteilung", f"H = {H_certain:.4f}")
        st.caption("[0.99, 0.01] — fast sicher")

    with col_comp2:
        p_balanced = np.array([0.5, 0.5])
        H_balanced = entropy(p_balanced)
        st.metric("Ausgewogen", f"H = {H_balanced:.4f}")
        st.caption("[0.5, 0.5] — maximale Unsicherheit")

    with col_comp3:
        p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
        H_uniform = entropy(p_uniform)
        st.metric("Gleichverteilung (4)", f"H = {H_uniform:.4f}")
        st.caption("[0.25, 0.25, 0.25, 0.25] — H_max = 2.0")

    st.subheader("Cross-Entropy & KL-Divergenz")
    st.markdown(r"""
    - **Cross-Entropy:** $H(P, Q) = -\sum_i p_i \cdot \log(q_i)$
      — wie gut approximiert Q die Verteilung P?
    - **KL-Divergenz:** $D_{KL}(P||Q) = \sum_i p_i \cdot \log\frac{p_i}{q_i}$
      — "Abstand" zwischen zwei Verteilungen

    Beides sind **Loss-Funktionen** im Machine Learning!
    """)

    if st.button("🔬 Cross-Entropy & KL-Divergenz berechnen"):
        p_true = np.array([1.0, 0.0])  # Wahre Klasse: 0
        q_preds = [
            ("Perfekte Vorhersage", np.array([0.99, 0.01])),
            ("Gute Vorhersage", np.array([0.8, 0.2])),
            ("Unsicher", np.array([0.5, 0.5])),
            ("Falsch", np.array([0.1, 0.9])),
        ]

        st.markdown("| Vorhersage-Qualität | Cross-Entropy | KL-Divergenz |")
        st.markdown("|-------------------|--------------|-------------|")
        for name, q in q_preds:
            ce = cross_entropy(p_true, q)
            kl = kl_divergence(p_true, q)
            st.markdown(f"| {name} | {ce:.4f} | {kl:.4f} |")

        st.info("""
        **Interpretation:**
        - Perfekte Vorhersage → Cross-Entropy ≈ 0
        - Falsche Vorhersage → Cross-Entropy sehr hoch
        - KL-Divergenz = Cross-Entropy - Entropie(P)
        """)

# ═══════════════════════════════════════════════════════════════
# Tab 3: Distanzmetriken
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("📏 Distanzmetriken")

    st.markdown("""
    Distanzmetriken sind fundamental für viele ML-Algorithmen (k-NN, Clustering, etc.).
    Unterschiedliche Metriken messen "Nähe" unterschiedlich.
    """)

    st.subheader("Interaktiver Distanz-Vergleich")

    col_dist1, col_dist2 = st.columns([1, 2])

    with col_dist1:
        st.markdown("**Punkt A**")
        a1 = st.slider("A₁", -5.0, 5.0, 1.0, 0.1, key="a1")
        a2 = st.slider("A₂", -5.0, 5.0, 0.0, 0.1, key="a2")
        st.markdown("**Punkt B**")
        b1 = st.slider("B₁", -5.0, 5.0, 0.0, 0.1, key="b1")
        b2 = st.slider("B₂", -5.0, 5.0, 1.0, 0.1, key="b2")

        a = np.array([a1, a2])
        b = np.array([b1, b2])

        eucl = euclidean_distance(a, b)
        cos_sim = cosine_similarity(a, b)
        cos_dist = cosine_distance(a, b)
        manhattan = np.sum(np.abs(a - b))

        st.metric("Euklidische Distanz", f"{eucl:.4f}")
        st.metric("Manhattan-Distanz", f"{manhattan:.4f}")
        st.metric("Cosinus-Ähnlichkeit", f"{cos_sim:.4f}")
        st.metric("Cosinus-Distanz", f"{cos_dist:.4f}")

    with col_dist2:
        fig_dist, ax_dist = plt.subplots(figsize=(7, 7))
        ax_dist.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
                       color='blue', width=0.015, label=f'A ({a[0]}, {a[1]})')
        ax_dist.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1,
                       color='red', width=0.015, label=f'B ({b[0]}, {b[1]})')

        # Euklidische Distanz als gestrichelte Linie
        ax_dist.plot([a[0], b[0]], [a[1], b[1]], 'k--', alpha=0.5, linewidth=1,
                     label=f'Euklid: {eucl:.2f}')

        # Manhattan-Distanz als Treppenlinie
        ax_dist.plot([a[0], a[0], b[0]], [a[1], b[1], b[1]], 'g--', alpha=0.5, linewidth=1,
                     label=f'Manhattan: {manhattan:.2f}')

        # Winkel für Cosinus
        angle_a = np.arctan2(a[1], a[0])
        angle_b = np.arctan2(b[1], b[0])
        arc = np.linspace(min(angle_a, angle_b), max(angle_a, angle_b), 50)
        r = 1.5
        ax_dist.plot(r * np.cos(arc), r * np.sin(arc), 'purple', linewidth=1.5,
                     label=f'Winkel: {np.abs(angle_a - angle_b):.2f} rad')

        ax_dist.set_xlim(-6, 6)
        ax_dist.set_ylim(-6, 6)
        ax_dist.axhline(0, color='gray', linewidth=0.5)
        ax_dist.axvline(0, color='gray', linewidth=0.5)
        ax_dist.set_aspect('equal')
        ax_dist.legend(loc='upper right')
        ax_dist.set_title('Vektoren & Distanzen', fontweight='bold')
        ax_dist.grid(True, alpha=0.3)
        st.pyplot(fig_dist)

    st.subheader("📖 Distanzmetriken im Vergleich")
    st.markdown("""
    | Metrik | Formel | Anwendung |
    |--------|--------|-----------|
    | **Euklidisch (L2)** | $\\sqrt{\\sum (a_i - b_i)^2}$ | k-NN, Clustering, "Luftlinie" |
    | **Manhattan (L1)** | $\\sum |a_i - b_i|$ | Hochdimensionale Daten, "Taxi-Distanz" |
    | **Cosinus-Ähnlichkeit** | $\\frac{a \\cdot b}{||a|| \\cdot ||b||}$ | Text-Ähnlichkeit |
    | **Cosinus-Distanz** | $1 - \\text{cosine\\_similarity}$ | Gleiche Anwendung wie Cosinus |
    """)

# ═══════════════════════════════════════════════════════════════
# Tab 4: Aktivierungsfunktionen
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("🧮 Aktivierungsfunktionen")

    st.markdown("""
    Aktivierungsfunktionen bringen **Nichtlinearität** in Neuronale Netze.
    Ohne sie wäre ein NN nur eine lineare Transformation!
    """)

    x_act = np.linspace(-5, 5, 200)

    fig_act, axes_act = plt.subplots(2, 2, figsize=(12, 10))

    # Sigmoid
    axes_act[0, 0].plot(x_act, sigmoid(x_act), 'b-', linewidth=2)
    axes_act[0, 0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes_act[0, 0].axhline(1, color='gray', linestyle='--', linewidth=0.5)
    axes_act[0, 0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes_act[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)', fontweight='bold')
    axes_act[0, 0].set_ylim(-0.1, 1.1)
    axes_act[0, 0].grid(True, alpha=0.3)

    # ReLU
    axes_act[0, 1].plot(x_act, np.maximum(0, x_act), 'r-', linewidth=2)
    axes_act[0, 1].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes_act[0, 1].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes_act[0, 1].set_title('ReLU: f(x) = max(0, x)', fontweight='bold')
    axes_act[0, 1].grid(True, alpha=0.3)

    # Softmax (2D Demo)
    x_soft = np.linspace(-2, 2, 50)
    X_soft, Y_soft = np.meshgrid(x_soft, x_soft)
    points = np.stack([X_soft.ravel(), Y_soft.ravel()], axis=1)
    sm = softmax(points)
    axes_act[1, 0].contourf(X_soft, Y_soft, sm[:, 0].reshape(50, 50), levels=20, cmap='RdYlBu')
    axes_act[1, 0].set_title('Softmax (2-Klassen): P(Klasse 1)', fontweight='bold')
    axes_act[1, 0].set_xlabel('Logit 1')
    axes_act[1, 0].set_ylabel('Logit 2')

    # Tanh
    axes_act[1, 1].plot(x_act, np.tanh(x_act), 'g-', linewidth=2)
    axes_act[1, 1].axhline(-1, color='gray', linestyle='--', linewidth=0.5)
    axes_act[1, 1].axhline(1, color='gray', linestyle='--', linewidth=0.5)
    axes_act[1, 1].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    axes_act[1, 1].set_title('Tanh: f(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)', fontweight='bold')
    axes_act[1, 1].set_ylim(-1.1, 1.1)
    axes_act[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_act)

    st.subheader("📖 Vergleich")
    st.markdown("""
    | Funktion | Bereich | Vorteil | Nachteil |
    |----------|---------|---------|----------|
    | **Sigmoid** | (0, 1) | Glatt, interpretierbar als Wahrscheinlichkeit | Vanishing Gradient |
    | **ReLU** | [0, ∞) | Kein Vanishing Gradient, schnell | "Dying ReLU" |
    | **Tanh** | (-1, 1) | Null-zentriert | Vanishing Gradient |
    | **Softmax** | (0, 1), Σ=1 | Wahrscheinlichkeitsverteilung | Nur für Output-Layer |
    """)

st.sidebar.markdown("""
### 📚 Über diese App

Interaktive Exploration der **mathematischen Grundlagen**
des Machine Learning.

**Funktionen:**
- 📉 Gradient Descent in 1D & 2D
- 🎲 Entropie, Cross-Entropy, KL-Divergenz
- 📏 Euklidisch, Manhattan, Cosinus
- 🧮 Sigmoid, ReLU, Tanh, Softmax

**Code:** `math_algorithms.py` im Repo
""")
