"""PCA/Eigenfaces + KNN, con control de rango y rechazo de desconocidos orientativo."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


@dataclass
class ResultadoPrediccion:
    etiqueta: str
    apoyo_vecinos: float
    distancia_vecino_mas_cercano: float
    vecinos: list[tuple[str, float]]
    coordenadas_pca: np.ndarray
    es_desconocido: bool = False


class ReconocedorEigenfaces:
    def __init__(self, tamano_imagen=(80, 80), componentes=12, vecinos=3):
        self.tamano_imagen = tuple(tamano_imagen)
        self.componentes_solicitados = int(componentes)
        self.vecinos_solicitados = int(vecinos)
        self.pca = None
        self.clasificador = None
        self.umbral_desconocido: float | None = None
        self.X = self.y = self.X_centrada = self.matriz_relacion = self.X_pca = None

    def entrenar(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
            raise ValueError('Datos de entrenamiento incorrectos (matriz X e identificadores y).')
        clases, conteos = np.unique(y, return_counts=True)
        if len(clases) < 2 or np.min(conteos) < 3:
            raise ValueError('Necesitas al menos 2 categorías y 3 imágenes diferentes de cada una.')
        if self.componentes_solicitados < 1 or self.vecinos_solicitados < 1:
            raise ValueError('Componentes y vecinos deben ser mayores que cero.')

        self.X = X
        self.y = y
        media = X.mean(axis=0)
        self.X_centrada = X - media
        self.matriz_relacion = self.X_centrada @ self.X_centrada.T / (len(X) - 1)
        espectro = np.linalg.eigvalsh(self.matriz_relacion)
        rango = np.sum(espectro > max(float(espectro[-1]), 1e-12) * 1e-8)
        if rango < 2:
            raise ValueError('Las imágenes tienen variación insuficiente para obtener componentes PCA útiles.')
        componentes = min(self.componentes_solicitados, len(X)-1, X.shape[1], int(rango))
        # whiten=False: evita amplificar ruido de componentes casi nulos en datasets pequeños.
        self.pca = PCA(n_components=componentes, whiten=False, svd_solver='randomized', random_state=42)
        self.X_pca = self.pca.fit_transform(X)
        self.clasificador = KNeighborsClassifier(
            n_neighbors=min(self.vecinos_solicitados, len(X)), weights='distance', metric='euclidean'
        )
        self.clasificador.fit(self.X_pca, y)
        self.umbral_desconocido = self._estimar_umbral()

    def _estimar_umbral(self) -> float | None:
        """Estimación de exploración; NO sustituye una validación con desconocidos reales."""
        dist = pairwise_distances(self.X_pca)
        np.fill_diagonal(dist, np.inf)
        positivos, negativos = [], []
        for i, etiqueta in enumerate(self.y):
            mismos = self.y == etiqueta
            diferentes = ~mismos
            if np.any(np.isfinite(dist[i, mismos])):
                positivos.append(float(np.min(dist[i, mismos])))
            if np.any(diferentes):
                negativos.append(float(np.min(dist[i, diferentes])))
        if not positivos or not negativos:
            return None
        positivos = np.asarray(positivos)
        negativos = np.asarray(negativos)
        max_bueno = float(np.percentile(positivos, 90)) * 1.30
        min_malo = float(np.percentile(negativos, 10)) * 0.85
        if max_bueno <= 0 or min_malo <= max_bueno:
            return None
        return (max_bueno + min_malo) / 2

    def predecir(self, vector, permitir_desconocido=True, umbral=None) -> ResultadoPrediccion:
        if self.pca is None or self.clasificador is None:
            raise RuntimeError('Entrena el modelo antes de predecir.')
        vector = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if vector.shape[1] != self.X.shape[1]:
            raise ValueError('La imagen no tiene el tamaño esperado.')
        coord = self.pca.transform(vector)
        etiqueta = str(self.clasificador.predict(coord)[0])
        probabilidades = self.clasificador.predict_proba(coord)[0]
        apoyo = float(np.max(probabilidades))
        distancias, indices = self.clasificador.kneighbors(coord, return_distance=True)
        vecinos = [(str(self.y[idx]), float(d)) for d, idx in zip(distancias[0], indices[0])]
        distancia = float(distancias[0][0])
        limite = self.umbral_desconocido if umbral is None else float(umbral)
        if limite is not None and limite <= 0:
            raise ValueError('El umbral debe ser un número positivo.')
        desconocido = bool(permitir_desconocido and limite is not None and distancia > limite)
        return ResultadoPrediccion('Desconocido' if desconocido else etiqueta, 0.0 if desconocido else apoyo,
                                   distancia, vecinos, coord[0], desconocido)

    @property
    def eigenfaces(self):
        if self.pca is None:
            raise RuntimeError('Primero entrena el modelo.')
        alto, ancho = self.tamano_imagen[1], self.tamano_imagen[0]
        return self.pca.components_.reshape((-1, alto, ancho))

    @property
    def rostro_promedio(self):
        if self.pca is None:
            raise RuntimeError('Primero entrena el modelo.')
        alto, ancho = self.tamano_imagen[1], self.tamano_imagen[0]
        return self.pca.mean_.reshape((alto, ancho))

    def resumen_matematico(self):
        if self.pca is None:
            raise RuntimeError('Primero entrena el modelo.')
        var = np.sum(self.pca.explained_variance_ratio_) * 100
        return ('MATRICES DEL MODELO PCA/EIGENFACES\n'
                f'X (muestras × píxeles): {self.X.shape}\n'
                f'X centrada: {self.X_centrada.shape}\n'
                f'Matriz de relación: {self.matriz_relacion.shape}\n'
                f'Componentes PCA: {self.pca.n_components_}\n'
                f'Varianza explicada: {var:.2f}%\n'
                f'Umbral desconocidos (heurístico): {self.umbral_desconocido}\n'
                f'Bloque de matriz:\n{self.matriz_relacion[:5,:5]}\n')
