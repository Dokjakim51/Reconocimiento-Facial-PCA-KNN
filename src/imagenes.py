"""Detección y preprocesamiento coherente de fotografías y video."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .calidad import listar_imagenes


def crear_detector_rostros() -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector.empty():
        raise RuntimeError('No se pudo abrir el detector facial Haar de OpenCV.')
    return detector


def detectar_rostro_principal(gris: np.ndarray, detector: cv2.CascadeClassifier):
    encontrados = detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))
    return max(encontrados, key=lambda r: int(r[2]) * int(r[3])) if len(encontrados) else None


def recortar_rostro(gris: np.ndarray, bbox, margen: float = .12) -> np.ndarray:
    x, y, w, h = map(int, bbox)
    mx, my = round(w * margen), round(h * margen)
    x1, y1 = max(0, x-mx), max(0, y-my)
    x2, y2 = min(gris.shape[1], x+w+mx), min(gris.shape[0], y+h+my)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('No se pudo recortar el rostro detectado.')
    return gris[y1:y2, x1:x2]


def preparar_rostro_desde_gris(gris: np.ndarray, tamano: tuple[int, int]):
    imagen = Image.fromarray(np.asarray(gris, dtype=np.uint8))
    imagen = ImageOps.autocontrast(imagen)
    imagen = ImageOps.fit(imagen, tamano, method=Image.Resampling.LANCZOS)
    matriz = np.asarray(imagen, dtype=np.float32) / 255.0
    return matriz, matriz.ravel()


def cargar_imagen_vector(ruta: Path, tamano: tuple[int, int], detector=None):
    with Image.open(ruta) as imagen:
        gris = np.asarray(ImageOps.exif_transpose(imagen).convert('L'), dtype=np.uint8)
    if detector is not None:
        bbox = detectar_rostro_principal(gris, detector)
        if bbox is None:
            raise ValueError(f'No se detectó un rostro en «{ruta.name}». Usa otra foto frontal.')
        gris = recortar_rostro(gris, bbox)
    return preparar_rostro_desde_gris(gris, tamano)


def cargar_dataset(carpeta: Path, tamano: tuple[int, int], detectar_rostros: bool = True):
    detector = crear_detector_rostros() if detectar_rostros else None
    vectores, etiquetas, rutas = [], [], []
    if not carpeta.exists():
        raise FileNotFoundError(f'No existe la carpeta: {carpeta}')
    for subcarpeta in sorted(x for x in carpeta.iterdir() if x.is_dir()):
        for ruta in listar_imagenes(subcarpeta):
            _, vector = cargar_imagen_vector(ruta, tamano, detector)
            vectores.append(vector)
            etiquetas.append(subcarpeta.name)
            rutas.append(ruta)
    if not vectores:
        raise ValueError('No se encontraron imágenes de entrenamiento.')
    return np.vstack(vectores), np.asarray(etiquetas), rutas


def cargar_pruebas(carpeta: Path, tamano: tuple[int, int], detectar_rostros=True):
    detector = crear_detector_rostros() if detectar_rostros else None
    return [(ruta, *cargar_imagen_vector(ruta, tamano, detector)) for ruta in listar_imagenes(carpeta)]
