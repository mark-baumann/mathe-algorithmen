# Mathematische Grundlagen des Maschinellen Lernens

Eine Reihe von fünf Jupyter-Notebooks, die die mathematischen Grundlagen des Maschinellen Lernens
in Lehrbuchform entwickeln: Definitionen, Sätze mit vollständigen Beweisen und anschließender
numerischer Verifikation der bewiesenen Aussagen.

Der Anspruch der Reihe ist, dass keine Behauptung unbelegt bleibt. Jeder Satz wird entweder
bewiesen oder mit Fundstelle zitiert, und jede Aussage über das numerische Verhalten eines
Verfahrens — Fehlerordnung, Konvergenzrate, Stabilität — wird im Notebook gemessen und mit dem
theoretischen Wert verglichen.

## Inhalt

| Nr. | Notebook | Themen | Umfang |
|-----|----------|--------|--------|
| 1 | [Vektorräume, Normen und Distanzmetriken](notebooks/01_vektorraeume_normen_und_distanzmetriken.ipynb) | Normaxiome, $p$-Normen, Metriken, Cauchy–Schwarz, Winkelgeometrie, Distanzkonzentration, Merkmalsskalierung | ~60 min |
| 2 | [Differentialrechnung und Gradientenverfahren](notebooks/02_differentialrechnung_und_gradientenverfahren.ipynb) | Gradient und Richtungsableitung, numerische Differentiation, Konvergenz des Gradientenverfahrens, $L$-Glattheit, Konditionszahl, Impulsverfahren | ~75 min |
| 3 | [Integralrechnung und numerische Quadratur](notebooks/03_integralrechnung_und_numerische_quadratur.ipynb) | Riemann-Integral, Hauptsatz, Newton–Cotes, Gauß–Legendre, adaptive Quadratur, Monte Carlo | ~75 min |
| 4 | [Wahrscheinlichkeit und Informationstheorie](notebooks/04_wahrscheinlichkeit_und_informationstheorie.ipynb) | Jensen-Ungleichung, Shannon-Entropie, Gibbs-Ungleichung, KL-Divergenz, Maximum-Likelihood, Transinformation | ~75 min |
| 5 | [Aktivierungsfunktionen und Nichtlinearität](notebooks/05_aktivierungsfunktionen_und_nichtlinearitaet.ipynb) | Universelle Approximation, Sigmoid und Tanh, ReLU-Varianten, Softmax-Jacobi-Matrix, Gradientenfluss, Initialisierung | ~70 min |

Die Notebooks bauen aufeinander auf und verweisen wechselseitig aufeinander — die
Cauchy–Schwarz-Ungleichung aus Notebook 1 begründet den steilsten Abstieg in Notebook 2, die
Konditionszahl aus Notebook 2 erklärt das Verhalten tiefer Netze in Notebook 5. Sie lassen sich
aber auch einzeln bearbeiten.

## Aufbau eines Notebooks

Alle fünf folgen derselben Gliederung:

1. **Kopf** — Zusammenfassung, Lernziele, Inhaltsverzeichnis mit Sprungmarken
2. **Vorbereitung** — identische Setup-Zelle: Importe, fester Zufallsseed, einheitliches
   Abbildungslayout
3. **Fachteil** — nummerierte Abschnitte im Wechsel aus Definition/Satz/Beweis und Codezelle,
   die die bewiesene Aussage numerisch prüft; jede Abbildung mit nummerierter Bildunterschrift
   und Interpretation
4. **Übungsaufgaben** — fünf Aufgaben je Notebook, vom Nachrechnen bis zur eigenständigen
   Erweiterung des Codes
5. **Literatur** — vollständige Angaben zu allen zitierten Quellen

Sätze sind fortlaufend nummeriert (`Satz 3.2` = Notebook-Abschnitt 3, zweiter Satz) und werden
notebookübergreifend referenziert.

## Reproduktion

Voraussetzung ist Python 3.10 oder neuer.

```bash
git clone https://github.com/mark-baumann/mathe-algorithmen.git
cd mathe-algorithmen

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab
```

Alle Notebooks laufen ohne weitere Konfiguration von oben nach unten durch. Sämtliche
stochastischen Experimente verwenden einen festen Seed, die Ausgaben sind daher exakt
reproduzierbar.

Der gesamte Bestand lässt sich nicht-interaktiv ausführen und dabei prüfen:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Der Befehl bricht bei der ersten fehlschlagenden Zelle ab und eignet sich damit als
Regressionstest der Reihe.

## Hinweise zur Fassung im Repository

- Die Notebooks sind **mit Ausgaben** eingecheckt, damit Text, Zahlenwerte und Abbildungen direkt
  auf GitHub lesbar sind, ohne dass eine Laufzeitumgebung nötig wäre.
- Der gesamte Code steht in den Notebooks selbst; es gibt bewusst kein importierbares
  Hilfsmodul. Jedes Notebook ist damit für sich vollständig und ohne Kenntnis der übrigen lesbar.
- Die Abbildungen verwenden durchgehend dieselbe, auf Farbfehlsichtigkeit geprüfte Farbfolge;
  jede Datenreihe ist zusätzlich über Legende, Linienstil oder Beschriftung identifizierbar.

## Abhängigkeiten

| Paket | Verwendung |
|-------|------------|
| NumPy | numerische Kernoperationen |
| SciPy | adaptive Quadratur, Legendre-Knoten, Fehlerfunktion |
| Matplotlib | sämtliche Abbildungen |
| JupyterLab | Ausführungsumgebung |

## Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann)
