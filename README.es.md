# Reconocimiento facial educativo con PCA (Eigenfaces) y KNN

Proyecto académico de **Inteligencia Artificial** creado con Python, OpenCV y scikit-learn. Explora el procesamiento de imágenes, la reducción de dimensiones mediante PCA y la clasificación KNN, con una interfaz gráfica y detección facial en vivo.

> **Este repositorio no incluye fotografías de personas ni bases de datos biométricas.** Para entrenarlo hay que agregar imágenes propias o autorizadas en el equipo local. Es una demostración educativa, **no** un sistema fiable de autenticación, vigilancia o control de acceso.

## Funcionalidades

- Detección de rostros por webcam, incluso sin fotografías de entrenamiento.
- Procesamiento de imágenes (recorte facial, escala de grises y tamaño 80 × 80).
- Clasificación experimental de rostros con **PCA/Eigenfaces + KNN** a partir de fotografías agregadas localmente.
- Evaluación con fotografías distintas de las usadas para entrenar: métricas y matriz de confusión.
- Avisos ante duplicados en entrenamiento y coincidencias entre entrenamiento y pruebas.
- Panel visual con cámara ajustable, pantalla completa (`F11`) y resultados locales.
- Pruebas automáticas con datos artificiales generados en memoria, sin compartir imágenes.

## Tecnologías

Python 3.12 · NumPy · scikit-learn · OpenCV · Pillow · Matplotlib · Tkinter

## Instalación (Windows, VS Code)

1. Descarga o clona este repositorio y abre la carpeta que contiene `main.py`.
2. En la terminal de PowerShell, ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. En VS Code, selecciona el intérprete `.venv\Scripts\python.exe` usando **Python: Select Interpreter**.
4. Abre `main.py` y pulsa **▶ Ejecutar archivo Python**.

**Primera ejecución:** el panel se abre sin datos de entrenamiento. Puedes pulsar **Probar cámara (detección)** para comprobar la webcam. **Reconocer (fotos locales)** requiere añadir imágenes primero.

## Cómo agregar fotografías de ejemplo

Las carpetas están inicialmente vacías a propósito. Crea las subcarpetas y coloca allí tus archivos locales:

```text
datos/
├── personas/              # Entrenamiento
│   ├── Persona_A/
│   │   ├── foto_01.jpg
│   │   ├── foto_02.jpg
│   │   └── ...
│   └── Persona_B/
│       ├── foto_01.jpg
│       ├── foto_02.jpg
│       └── ...
└── prueba/                # Evaluación: fotos DIFERENTES a las anteriores
    ├── Persona_A/
    │   └── foto_nueva_01.jpg
    └── Persona_B/
        └── foto_nueva_01.jpg
```

- Mínimo: **2 personas distintas** y **3 fotografías diferentes por persona** en `datos/personas/`.
- Recomendado para explorar variaciones: **8 a 15** fotografías distintas por persona y **3 a 5 nuevas** para la evaluación.
- Usa fotografías frontales con iluminación suficiente. Si la cara no es detectable, el programa te avisará.
- Pide autorización explícita a cada participante; utiliza etiquetas como `Persona_A` en vez de nombres completos.
- **Nunca copies las mismas fotos en las carpetas de entrenamiento y prueba.** Cambiarles el nombre tampoco las convierte en fotografías nuevas.
- **No subas las fotos a GitHub.** El `.gitignore` las excluye del repositorio, pero verifica siempre qué se va a publicar.

Una vez agregadas, **vuelve a pulsar ▶ Ejecutar**. Si tienes fotos de prueba y entrenamiento suficientes, se mostrará la evaluación y podrás abrir **Reconocer (fotos locales)** para explorar coincidencias tentativas mediante la webcam.

Si solo tienes imágenes para entrenamiento, aparecerán las instrucciones en la ventana; también puedes pulsar **Reconocer (fotos locales)** y entrenar sin evaluación, aunque se recomienda añadir imágenes de prueba independientes.

## Ejecutar desde la terminal (opcional)

```powershell
# Revisar si hay duplicados o faltan datos
.\.venv\Scripts\python.exe main.py --modo diagnostico

# Entrenar, evaluar y calcular métricas con fotos nuevas
.\.venv\Scripts\python.exe main.py --modo evaluar --graficas

# Reconocimiento experimental en ventana OpenCV (Q o ESC para salir)
.\.venv\Scripts\python.exe main.py --modo camara

# Pruebas automáticas sin fotografías personales
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Para volver al panel visual desde PowerShell: `python main.py --ventana` utilizando el entorno virtual.

## Estructura

```text
Reconocimiento_Facial_PCA/
├── main.py
├── src/
│   ├── calidad.py          # Auditoría de conjuntos de imágenes
│   ├── imagenes.py         # Detección y preparación
│   ├── modelo.py           # PCA y KNN
│   ├── camara.py           # Webcam con reconocimiento experimental
│   └── visualizacion.py    # Resultados, gráficos y Eigenfaces
├── datos/
│   ├── personas/.gitkeep   # Sin fotografías incluidas
│   └── prueba/.gitkeep
├── tests/
│   └── test_proyecto.py
├── docs/
│   ├── GUIA_PARA_FOTOS.md
│   └── PLAN_DE_VALIDACION.md
├── requirements.txt
├── .gitignore
└── .github/workflows/tests.yml
```

## Privacidad y limitaciones

- Las imágenes se procesan localmente. La cámara no graba video por defecto.
- `datos/`, `resultados/` y los archivos de modelos locales están excluidos del seguimiento de Git.
- Los valores de apoyo de KNN **no son probabilidades de identidad**.
- La detección o clasificación puede fallar por iluminación, pose, expresiones y calidad de cámara.
- Los resultados de las pruebas automáticas **no representan exactitud de reconocimiento facial real**. La exactitud real todavía debe evaluarse con fotografías independientes y autorizadas.
- No emplees el proyecto para verificar identidades, conceder acceso ni tomar decisiones sobre otras personas.

Consulta [Guía de imágenes](docs/GUIA_PARA_FOTOS.md), [Plan de validación](docs/PLAN_DE_VALIDACION.md) y [Guía para publicar en GitHub](docs/PUBLICACION_GITHUB.md).

**Autor:** Elian Girón. Proyecto académico de Desarrollo en Inteligencia Artificial.
