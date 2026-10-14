"""Auditoría de datos: detectar duplicaciones y filtraciones entre entrenamiento y prueba.

Se comparan los bytes (SHA-256) y el contenido de imagen normalizado (SHA-256
sobre píxeles RGB); así se detectan también copias idénticas recodificadas.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from PIL import Image, ImageOps

EXTENSIONES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def listar_imagenes(carpeta: Path) -> list[Path]:
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONES)


def _huella_imagen(ruta: Path) -> str:
    # Recodificaciones que conservan píxeles idénticos generan la misma huella.
    with Image.open(ruta) as img:
        rgb = ImageOps.exif_transpose(img).convert('RGB')
        return sha256(f'{rgb.size}'.encode('ascii') + rgb.tobytes()).hexdigest()


@dataclass
class Auditoria:
    entrenamiento: int
    prueba: int
    unicas: dict[str, int]
    repetidas_entrenamiento: list[list[str]]
    filtraciones: list[list[str]]
    repetidas_prueba: list[list[str]]
    errores: list[str]
    advertencias: list[str]

    @property
    def valido(self) -> bool:
        return not self.errores

    def mostrar(self) -> str:
        lineas = [
            '\n=== AUDITORÍA DE DATOS ===',
            f'Archivos entrenamiento: {self.entrenamiento}',
            f'Archivos prueba: {self.prueba}',
            'Imágenes diferentes en entrenamiento por categoría: ' +
            (', '.join(f'{x}={n}' for x, n in sorted(self.unicas.items())) or 'NINGUNA'),
            f'Grupos repetidos en entrenamiento: {len(self.repetidas_entrenamiento)}',
            f'Grupos repetidos en prueba: {len(self.repetidas_prueba)}',
            f'Filtraciones entrenamiento ↔ prueba: {len(self.filtraciones)}',
        ]
        for err in self.errores:
            lineas.append('ERROR: ' + err)
        for adv in self.advertencias:
            lineas.append('AVISO: ' + adv)
        return '\n'.join(lineas)


def auditar_datos(carpeta_entrenamiento: Path, carpeta_prueba: Path) -> Auditoria:
    train = listar_imagenes(carpeta_entrenamiento)
    test = listar_imagenes(carpeta_prueba)
    grupos_train: dict[str, list[Path]] = defaultdict(list)
    grupos_test: dict[str, list[Path]] = defaultdict(list)
    errores, advertencias = [], []

    for ruta in train:
        try:
            grupos_train[_huella_imagen(ruta)].append(ruta)
        except (OSError, ValueError) as exc:
            errores.append(f'Imagen de entrenamiento ilegible: {ruta.name}: {exc}')
    for ruta in test:
        try:
            grupos_test[_huella_imagen(ruta)].append(ruta)
        except (OSError, ValueError) as exc:
            errores.append(f'Imagen de prueba ilegible: {ruta.name}: {exc}')

    unicas: dict[str, int] = defaultdict(int)
    for rutas in grupos_train.values():
        # El mismo archivo bajo dos etiquetas es un error adicional.
        etiquetas = {r.parent.name for r in rutas}
        for etiqueta in etiquetas:
            unicas[etiqueta] += 1
        if len(etiquetas) > 1:
            errores.append('La misma fotografía está asignada a distintas personas: ' + ', '.join(sorted(etiquetas)))

    repetidas_train = [[str(p.relative_to(carpeta_entrenamiento)) for p in rutas]
                       for rutas in grupos_train.values() if len(rutas) > 1]
    repetidas_test = [[str(p.relative_to(carpeta_prueba)) for p in rutas]
                      for rutas in grupos_test.values() if len(rutas) > 1]
    filtraciones = [
        [*('entrenamiento/' + str(p.relative_to(carpeta_entrenamiento)) for p in grupos_train[h]),
         *('prueba/' + str(p.relative_to(carpeta_prueba)) for p in grupos_test[h])]
        for h in (grupos_train.keys() & grupos_test.keys())
    ]

    if not train:
        errores.append('Faltan imágenes de entrenamiento en datos/personas/NOMBRE/.')
    if len(unicas) < 2:
        errores.append('Se necesitan al menos dos categorías diferentes de entrenamiento.')
    for etiqueta, cantidad in sorted(unicas.items()):
        if cantidad < 3:
            errores.append(f'{etiqueta}: solo {cantidad} imagen(es) diferente(s); mínimo 3, recomendado 8 o más.')
    if repetidas_train:
        errores.append('Hay fotografías repetidas en entrenamiento. Elimina las copias.')
    if filtraciones:
        errores.append('La misma imagen aparece en entrenamiento y prueba. Sustituye las pruebas por fotos nuevas.')
    if repetidas_test:
        advertencias.append('Hay fotografías repetidas dentro de prueba; no cuentes esos duplicados como evaluaciones independientes.')
    if not test:
        advertencias.append('No hay imágenes de prueba independiente; no se puede medir exactitud real.')
    return Auditoria(len(train), len(test), dict(unicas), repetidas_train, filtraciones,
                    repetidas_test, errores, advertencias)
