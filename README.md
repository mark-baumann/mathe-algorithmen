# 📐 Mathematische Algorithmen für Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Aktiv-brightgreen.svg)]()

Interaktive Exploration der **mathematischen Grundlagen des Machine Learning**. Gradient Descent in 1D und 2D animieren, Entropie und Informationstheorie verstehen, Distanzmetriken vergleichen und Aktivierungsfunktionen visualisieren — alles live in einer Streamlit-App mit anpassbaren Parametern.

## ✨ Features

- **📉 Gradient Descent** — 1D und 2D mit animierten Pfaden, Learning Rate experimentell erkunden
- **🎲 Entropie & Informationstheorie** — Shannon-Entropie, Cross-Entropy und KL-Divergenz berechnen und vergleichen
- **📏 Distanzmetriken** — Euklidisch, Manhattan und Cosinus-Ähnlichkeit interaktiv visualisieren
- **🧮 Aktivierungsfunktionen** — Sigmoid, ReLU, Tanh und Softmax mit ihren Eigenschaften
- **🎛️ Interaktive Regler** — Alle Parameter live anpassen und Effekte sofort sehen
- **✅ Vollständig getestet** — Unit-Tests für alle mathematischen Funktionen

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/mark-baumann/mathe-algorithmen.git
cd mathe-algorithmen

# Virtuelle Umgebung erstellen
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Abhängigkeiten installieren
uv pip install -e ".[dev]"
```

## 🎯 Nutzung

```bash
# Streamlit-App starten
streamlit run app.py
```

Die App öffnet sich im Browser unter `http://localhost:8501`. Wähle einen der vier Tabs und experimentiere mit den Parametern.

## 🧪 Tests ausführen

```bash
pytest tests/ -v
```

## 🛠️ Tech-Stack

| Technologie | Einsatz |
|-------------|---------|
| **NumPy** | Mathematische Kernoperationen |
| **Matplotlib** | Visualisierung von Funktionen, Gradienten und Verteilungen |
| **Streamlit** | Interaktive Web-App |
| **Pytest** | Test-Framework |
| **Ruff** | Linting & Code-Qualität |

## 📁 Projektstruktur

```
mathe-algorithmen/
├── app.py                  # Streamlit-Hauptapp (4 Tabs)
├── pyproject.toml          # Projekt-Konfiguration
├── math_algorithms.py      # Kern-Implementierung aller Algorithmen
└── tests/
    ├── __init__.py
    └── test_math_algorithms.py
```

## 📖 Enthaltene Algorithmen

| Algorithmus | Formel | Anwendung |
|-------------|--------|-----------|
| **Gradient Descent** | $x_{t+1} = x_t - \eta \nabla f(x_t)$ | Optimierung, Training von NNs |
| **Entropie** | $H(P) = -\sum p_i \log_2(p_i)$ | Entscheidungsbäume, Informationsgewinn |
| **Cross-Entropy** | $H(P,Q) = -\sum p_i \log(q_i)$ | Loss-Funktion für Klassifikation |
| **KL-Divergenz** | $D_{KL}(P\|\|Q) = \sum p_i \log\frac{p_i}{q_i}$ | Modellvergleich, VAEs |
| **Euklidische Distanz** | $\sqrt{\sum (a_i - b_i)^2}$ | k-NN, Clustering |
| **Cosinus-Ähnlichkeit** | $\frac{a \cdot b}{\|\|a\|\| \cdot \|\|b\|\|}$ | Textähnlichkeit, Embeddings |
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Logistic Regression, Output-Layer |
| **Softmax** | $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$ | Multi-Klassen-Klassifikation |

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)

---

*Mathematik ist die Sprache des Machine Learning. Diese App macht abstrakte Konzepte durch interaktive Visualisierung greifbar.*
