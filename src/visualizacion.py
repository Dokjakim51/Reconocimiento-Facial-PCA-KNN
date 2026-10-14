from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .modelo import ReconocedorEigenfaces, ResultadoPrediccion


def guardar_eigenfaces(modelo: ReconocedorEigenfaces, salida: Path, max_eigenfaces: int = 8) -> None:
    eigenfaces = modelo.eigenfaces
    n = min(max_eigenfaces, len(eigenfaces))

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    axes[0].imshow(modelo.rostro_promedio, cmap='gray')
    axes[0].set_title('Rostro promedio')
    axes[0].axis('off')

    for i in range(n):
        ax = axes[i + 1]
        ax.imshow(eigenfaces[i], cmap='gray')
        ax.set_title(f'Eigenface {i + 1}')
        ax.axis('off')

    for ax in axes[n + 1:]:
        ax.axis('off')

    fig.suptitle('PCA / Eigenfaces obtenidas del conjunto de entrenamiento')
    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=160, bbox_inches='tight')
    plt.close(fig)


def guardar_resultado(
    matriz: np.ndarray,
    nombre_archivo: str,
    resultado: ResultadoPrediccion,
    salida: Path,
    mostrar: bool = False,
) -> None:
    fig = plt.figure(figsize=(9, 5.4))
    ax1 = fig.add_axes([0.06, 0.11, 0.42, 0.68])
    ax2 = fig.add_axes([0.53, 0.11, 0.42, 0.68])

    ax1.imshow(matriz, cmap='gray', vmin=0, vmax=1)
    ax1.set_title(f'Imagen de prueba\n{nombre_archivo}')
    ax1.axis('off')

    ax2.axis('off')
    vecinos = '\n'.join(
        f'• {etiqueta}: distancia {dist:.3f}'
        for etiqueta, dist in resultado.vecinos
    )
    texto = (
        f'PERSONA RECONOCIDA\n\n'
        f'{resultado.etiqueta}\n\n'
        f'Apoyo de vecinos KNN (no es probabilidad): '
        f'{resultado.apoyo_vecinos * 100:.1f}%\n\n'
        f'Vecinos usados para comparar:\n{vecinos}\n\n'
        'Aviso: el apoyo KNN NO es precision ni probabilidad de identidad.'
    )
    ax2.text(0.02, 0.98, texto, va='top', ha='left', fontsize=11)

    fig.suptitle('Reconocimiento facial con PCA (Eigenfaces) + KNN', y=0.97)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=160, bbox_inches='tight')

    if mostrar:
        plt.show()
    plt.close(fig)


def guardar_resumen_pruebas(items, salida: Path, mostrar: bool = False) -> None:
    """Guarda una sola figura con todas las pruebas para facilitar la exposición."""
    if not items:
        return
    n = len(items)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (matriz, nombre_archivo, resultado) in zip(axes, items):
        ax.imshow(matriz, cmap='gray', vmin=0, vmax=1)
        ax.set_title(
            f'{nombre_archivo}\n→ {resultado.etiqueta} '
            f'({resultado.apoyo_vecinos*100:.1f}%)',
            fontsize=10,
        )
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle('Resultados de reconocimiento facial — PCA / Eigenfaces + KNN', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=160, bbox_inches='tight')
    if mostrar:
        plt.show()
    plt.close(fig)
