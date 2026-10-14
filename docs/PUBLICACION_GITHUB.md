# Publicación del proyecto en GitHub (cuando estés listo)

Esta es la guía para **publicar más adelante**, no una publicación automática.

## 1. Datos preparados para el repositorio

| Campo de GitHub | Contenido sugerido |
|---|---|
| Nombre | `Reconocimiento-Facial-PCA-KNN` |
| Visibilidad | `Public` (si deseas usarlo en tu portafolio) |
| Descripción corta | Educational face recognition project using PCA (Eigenfaces), KNN and OpenCV, with webcam detection and a Tkinter GUI. |
| Topics | `python`, `machine-learning`, `computer-vision`, `pca`, `eigenfaces`, `knn`, `opencv`, `tkinter`, `artificial-intelligence` |
| Página web | Déjala vacía si no tienes una demo pública. No inventes un enlace. |

**Importante:** el programa es educativo y **no** un sistema de autenticación ni reconocimiento de identidad fiable. No prometas porcentajes de exactitud con personas reales que todavía no hayas medido.

## 2. Antes de subir los archivos

- [ ] Descomprime el ZIP final y abre la carpeta que contiene `main.py`.
- [ ] Comprueba que `datos/personas/` y `datos/prueba/` solo contienen `.gitkeep`.
- [ ] Comprueba que no haya fotografías, videos, datos biométricos, contraseñas, rutas privadas, entorno `.venv` ni resultados de experimentos.
- [ ] Comprueba que aparecen `main.py`, `src/`, `tests/`, `docs/`, `.github/`, `.gitignore`, `.gitattributes`, `README.md`, `README.es.md` y `requirements.txt`.
- [ ] Conserva una copia ZIP en tu computadora antes de publicar.

## 3. Método sencillo: subir desde la página de GitHub

1. Entra a <https://github.com/new> e inicia sesión.
2. Completa el nombre, descripción y visibilidad indicados arriba.
3. **No marques** «Add a README file» ni generes un `.gitignore` adicional, porque ya están incluidos.
4. Crea el repositorio.
5. Selecciona **Add file > Upload files** y sube **el contenido** de la carpeta extraída, no el ZIP en sí.
6. Comprueba que GitHub recibió todos los archivos y carpetas, incluidas `.github`, `.gitignore` y `.gitattributes`.
7. Escribe un mensaje como `Publicación inicial del proyecto PCA + KNN` y confirma los archivos.
8. En **About**, agrega los *topics* de la tabla.
9. Comprueba que el README aparece en la página principal y revisa la pestaña **Actions** para el resultado de las pruebas automáticas.

## 4. Método alternativo: terminal de VS Code

Ejecuta esto desde la **carpeta que contiene `main.py`** con el nombre de tu cuenta ya configurado:

```powershell
git init
git add .
git status --short
# Revisa que NO aparezcan fotos, .venv ni resultados antes del commit.
git commit -m "Publicación inicial del proyecto PCA + KNN"
git branch -M main
git remote add origin https://github.com/Dokjakim51/Reconocimiento-Facial-PCA-KNN.git
git push -u origin main
```

Si Git te pide configurar nombre o correo, hazlo con los datos que quieras mostrar como autor de los commits. Para subir, GitHub puede pedir autorización en el navegador. **No compartas tokens ni contraseñas**.

## 5. Después de publicar

- [ ] Revisa que el repositorio sea accesible y que no haya datos personales.
- [ ] Confirma que **Actions** pase las pruebas en GitHub. Las 9 pruebas locales no garantizan que todas las instalaciones se comporten igual.
- [ ] Añade el enlace del repositorio a tu currículum o perfil de LinkedIn.
- [ ] Si más adelante obtienes resultados con fotografías autorizadas, documenta el tamaño del conjunto de prueba y sus limitaciones sin publicar las fotos de terceros.
- [ ] Decide si quieres añadir una licencia de código abierto (por ejemplo, MIT); publicar sin una licencia no concede automáticamente permiso para reutilizar el código.

## 6. Texto breve para el currículum

**Reconocimiento Facial con PCA y KNN | Python**  
Proyecto académico de visión por computadora con detección facial, clasificación experimental mediante Eigenfaces y KNN, interfaz gráfica y pruebas automatizadas.

No lo describas como un sistema de seguridad ni como un modelo validado con un porcentaje de precisión real, porque esa comprobación está pendiente.
