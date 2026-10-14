from __future__ import annotations

import os
import time
from collections import Counter, deque

import cv2

from .imagenes import crear_detector_rostros, detectar_rostro_principal, preparar_rostro_desde_gris, recortar_rostro
from .modelo import ReconocedorEigenfaces, ResultadoPrediccion


def _abrir_camara(indice: int) -> cv2.VideoCapture:
    """Intenta abrir la webcam de forma amigable en Windows y otros sistemas."""
    if os.name == 'nt' and hasattr(cv2, 'CAP_DSHOW'):
        cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(indice)


def ejecutar_camara(
    modelo: ReconocedorEigenfaces,
    indice_camara: int = 0,
    umbral: float | None = None,
) -> int:
    detector = crear_detector_rostros()
    cap = _abrir_camara(indice_camara)

    if not cap.isOpened():
        print('\n[ERROR] No se pudo abrir la cámara.')
        print('Comprueba que ninguna otra aplicación la esté usando y que Windows permita el acceso a la cámara.')
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    print('\n=== MODO CÁMARA ===')
    print('Mira hacia la cámara con buena iluminación.')
    print('Presiona Q o ESC para cerrar la cámara.')
    print('Solo utiliza fotografías de personas que hayan autorizado las pruebas.')
    print('El apoyo KNN NO representa una probabilidad de identificación.')
    if modelo.umbral_desconocido is None and umbral is None:
        print('[AVISO] No hay umbral confiable para desconocidos: el modelo siempre propondrá alguna categoría.')

    historial: deque[str] = deque(maxlen=6)
    ultimo_resultado: ResultadoPrediccion | None = None
    ultimo_tiempo = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('[ERROR] No se pudo leer un fotograma de la cámara.')
                break

            frame = cv2.flip(frame, 1)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostro = detectar_rostro_principal(gris, detector)

            if rostro is not None:
                x, y, w, h = rostro
                roi = recortar_rostro(gris, rostro)

                # No hace falta predecir 30 veces por segundo. Esto estabiliza la imagen
                # y reduce carga de CPU sin afectar la demostración en vivo.
                ahora = time.monotonic()
                if ahora - ultimo_tiempo >= 0.12:
                    _, vector = preparar_rostro_desde_gris(roi, modelo.tamano_imagen)
                    ultimo_resultado = modelo.predecir(
                        vector,
                        permitir_desconocido=True,
                        umbral=umbral,
                    )
                    historial.append(ultimo_resultado.etiqueta)
                    ultimo_tiempo = ahora

                etiqueta = 'Analizando...'
                confianza = 0.0
                distancia = 0.0

                if ultimo_resultado is not None:
                    distancia = ultimo_resultado.distancia_vecino_mas_cercano
                    if historial:
                        etiqueta = Counter(historial).most_common(1)[0][0]
                    else:
                        etiqueta = ultimo_resultado.etiqueta
                    confianza = ultimo_resultado.apoyo_vecinos

                desconocido = etiqueta == 'Desconocido'
                color = (0, 0, 255) if desconocido else (0, 200, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                texto = etiqueta
                if etiqueta not in {'Desconocido', 'Analizando...'}:
                    texto += f'  apoyo KNN: {confianza * 100:.0f}%' 
                cv2.putText(
                    frame,
                    texto,
                    (x, max(28, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f'Distancia PCA: {distancia:.2f}',
                    (x, min(frame.shape[0] - 15, y + h + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                historial.clear()
                ultimo_resultado = None
                cv2.putText(
                    frame,
                    'No se detecta un rostro',
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                'PCA / Eigenfaces + KNN | Q o ESC para cerrar camara',
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow('Reconocimiento facial en vivo - PCA / Eigenfaces', frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord('q'), ord('Q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print('Cámara cerrada correctamente.')
    return 0
