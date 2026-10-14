# Face Recognition with PCA (Eigenfaces) + KNN

**Educational computer vision project** built with Python, OpenCV and scikit-learn. It explores facial detection, Principal Component Analysis (Eigenfaces) for dimensionality reduction, and K-Nearest Neighbors (KNN) for experimental image classification.

**[Leer en español](README.es.md)**

> **Privacy first:** This repository contains **no photos of real people, face datasets or trained biometric models**. The project starts with empty data folders. For experiments, supply your own images locally with appropriate consent. It is **not a secure identity verification, surveillance, or access-control system**.

## Highlights

- **Webcam face detection:** OpenCV-based facial detection works without any training photos.
- **PCA + KNN pipeline:** Crops and normalizes grayscale face images (80 × 80), computes principal components, and classifies projected samples with KNN.
- **Desktop GUI:** A Tkinter interface for webcam detection, local recognition experiments and evaluation results.
- **Dataset checks:** Finds duplicate training images and data leakage between training and evaluation folders.
- **Evaluation workflow:** Displays a confusion matrix and classification metrics for separate local test photos.
- **Automated tests:** Validates the model and data-handling logic with synthetic arrays, without needing photos or a webcam.

## Tech stack

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Computer vision | OpenCV |
| Machine learning | scikit-learn (PCA, KNN) |
| Numerical and imaging tools | NumPy, Pillow, Matplotlib |
| Desktop interface | Tkinter |
| Testing | `unittest`, GitHub Actions |

## Getting started (Windows / VS Code)

**Requirements:** Python 3.12 and a webcam for live detection. The Tkinter GUI requires a desktop environment.

1. Download the repository and open the folder containing `main.py` in VS Code.
2. Open the VS Code **PowerShell** terminal and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. In VS Code, run **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`. This also makes the ▶ Run button use the correct Python environment.
4. Run `main.py` (▶) to open the GUI. Without images, it shows an introductory screen and lets you try the **face-detection-only** webcam mode.

**Without a GUI** you can run the following diagnostics and tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py --modo diagnostico
```

The diagnostic command reports missing training data until you add your own images. That is **expected** in this public version; it is not a broken installation.

## Optional: local recognition experiments

The public repository starts with these intentionally empty directories:

```text
datos/
  personas/
    Persona_A/                  # Create locally; put training photos here
    Persona_B/
  prueba/
    Persona_A/                  # New, independent photos for evaluation
    Persona_B/
```

- At least **two classes** and **three different training images per class** are required by the model. For meaningful evaluation, collect more diverse images and keep test photos completely separate.
- **Use only photographs you have permission to process** and keep them on your machine. Use aliases rather than identifying folder names.
- The dataset checker rejects duplicate training photos and matches between training and test sets. Renaming copied photos does not make them independent observations.
- **Do not upload local face images, trained models or evaluation outputs to GitHub.** The `.gitignore` helps when using Git, but does **not** prevent accidental uploads through the GitHub website.

After preparing the images locally, run the GUI again or evaluate via PowerShell:

```powershell
.\.venv\Scripts\python.exe main.py --modo evaluar --graficas
```

This may save sensitive visualizations to the local `resultados/` directory; do not commit those outputs.

For additional details, see [Photo preparation guide](docs/GUIA_PARA_FOTOS.md) and [Validation plan](docs/PLAN_DE_VALIDACION.md).

## Project structure

```text
.
├── main.py                       # CLI and Tkinter GUI
├── src/
│   ├── calidad.py                # Dataset audit (duplicates / leakage)
│   ├── imagenes.py               # Face detection and preprocessing
│   ├── modelo.py                 # PCA + KNN learning and predictions
│   ├── camara.py                 # Live webcam mode
│   └── visualizacion.py          # Results and plots
├── datos/
│   ├── personas/.gitkeep         # Empty; private photos stay local
│   └── prueba/.gitkeep           # Empty; private test photos stay local
├── tests/                        # Synthetic automated tests
├── docs/                         # Documentation and safety guidance
├── requirements.txt
├── .gitignore
└── .github/workflows/tests.yml   # CI
```

## What the tests do and do not prove

Local automated checks cover data auditing, PCA/KNN prediction behavior, preprocessing and clean first launch. **Passing those tests does not establish real-world face-recognition accuracy.** No benchmark on independent, consented real-face photographs is included in this repository.

KNN neighbor scores are **not calibrated probabilities of identity**, and the unknown-person threshold is only heuristic. Lighting, camera quality, image overlap and other conditions can cause failures. **Do not use this code to authenticate people or make consequential decisions about them.**

## Author and status

Academic project by **Elian Girón**, AI Development student at ITLA (Dominican Republic).

- **Status:** Educational prototype; automated checks available.
- **Real-world recognition accuracy:** Not established in this public dataset-free release.
- **License:** No open-source license is provided at this time. Please ask before reusing the code.

[Publication checklist (Spanish)](docs/PUBLICACION_GITHUB.md)
