"""Demostración educativa de detección facial y clasificación PCA + KNN.

Las fotografías se mantienen fuera del repositorio y se procesan localmente.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.calidad import auditar_datos, listar_imagenes
from src.camara import ejecutar_camara
from src.imagenes import (cargar_dataset, cargar_imagen_vector, crear_detector_rostros,
                         recortar_rostro, preparar_rostro_desde_gris)
from src.modelo import ReconocedorEigenfaces
from src.visualizacion import guardar_eigenfaces, guardar_resultado, guardar_resumen_pruebas

BASE = Path(__file__).resolve().parent
DATOS_REALES = BASE / 'datos' 


def parsear_argumentos(argv=None):
    p = argparse.ArgumentParser(description='Eigenfaces (PCA) + KNN: proyecto educativo, no autenticación segura.')
    p.add_argument('--modo', choices=('menu', 'diagnostico', 'imagenes', 'evaluar', 'camara'), default='evaluar',
                   help='Por defecto abre el panel visual y evalúa fotos locales si existen.')
    p.add_argument('--datos', type=Path, default=DATOS_REALES,
                   help='Carpeta local con personas/ y prueba/. Por defecto: datos/.')
    p.add_argument('--imagen', type=Path, help='Analizar una fotografía particular (modo imágenes).')
    p.add_argument('--camara', type=int, default=0)
    p.add_argument('--componentes', type=int, default=12)
    p.add_argument('--vecinos', type=int, default=3)
    p.add_argument('--tamano', nargs=2, type=int, metavar=('ANCHO', 'ALTO'), default=(80, 80))
    p.add_argument('--umbral', type=float, help='Distancia máxima PCA para aceptar un rostro conocido. Requiere calibración.')
    p.add_argument('--graficas', action='store_true', help='Guarda figuras e imágenes de prueba SOLO en el equipo local.')
    p.add_argument('--ventana', action='store_true', help='Abre el panel visual de pruebas.')
    args = p.parse_args(argv)
    # Sin parámetros: VS Code abre la interfaz y evalúa fotos locales si hay datos suficientes.
    parametros = sys.argv[1:] if argv is None else argv
    if not parametros:
        args.ventana = True
        args.graficas = True
    elif args.ventana:
        args.graficas = True
    return args


def carpeta_datos(args):
    return args.datos.expanduser().resolve()


def auditar(args):
    raiz = carpeta_datos(args)
    datos = auditar_datos(raiz / 'personas', raiz / 'prueba')
    print(datos.mostrar())
    return datos


def entrenar(args):
    revision = auditar(args)
    if not revision.valido:
        raise ValueError('Corrige los problemas de los datos mostrados antes de entrenar.')
    raiz = carpeta_datos(args)
    X, y, _ = cargar_dataset(raiz / 'personas', tuple(args.tamano), detectar_rostros=True)
    modelo = ReconocedorEigenfaces(tuple(args.tamano), args.componentes, args.vecinos)
    modelo.entrenar(X, y)
    print('\nModelo entrenado:')
    print(f' Muestras={len(X)} | clases={len(set(y))} | PCA={modelo.pca.n_components_} componentes | KNN={modelo.clasificador.n_neighbors}')
    print(' IMPORTANTE: ningún porcentaje de vecinos KNN indica exactitud ni probabilidad de identidad.')
    if modelo.umbral_desconocido is None and args.umbral is None:
        print(' AVISO: sin umbral válido para rechazar personas desconocidas; usa datos mejores o valida uno manual.')
    return modelo


def rutas_prueba(args, para_evaluar=False):
    raiz = carpeta_datos(args) / 'prueba'
    if args.imagen is not None:
        return [args.imagen.expanduser().resolve()]
    rutas = listar_imagenes(raiz)
    if para_evaluar:
        rutas = [p for p in rutas if p.parent != raiz]
    return rutas


def ejecutar_imagenes(args, modelo, evaluar=False):
    rutas = rutas_prueba(args, para_evaluar=evaluar)
    if not rutas:
        raise ValueError('No hay imágenes. Para evaluar, ubícalas en datos/prueba/NOMBRE/*.jpg.')
    detector = crear_detector_rostros()
    root_test = carpeta_datos(args) / 'prueba'
    resultados, verdad, predicciones, graficas = [], [], [], []
    for ruta in rutas:
        matriz, vector = cargar_imagen_vector(ruta, tuple(args.tamano), detector)
        res = modelo.predecir(vector, permitir_desconocido=True, umbral=args.umbral)
        esperada = ruta.parent.name if ruta.parent != root_test and args.imagen is None else None
        if evaluar:
            if esperada is None:
                raise ValueError('La evaluación exige datos/prueba/NOMBRE/foto.jpg (nombres de categoría).')
            verdad.append(esperada)
            predicciones.append(res.etiqueta)
        print(f'{ruta.name}: {res.etiqueta} | apoyo KNN={res.apoyo_vecinos:.2f} | distancia PCA={res.distancia_vecino_mas_cercano:.3f}'
              + (f' | etiqueta esperada={esperada}' if evaluar else ''))
        resultados.append({
            'archivo': ruta.name, 'etiqueta_esperada': esperada, 'prediccion': res.etiqueta,
            'apoyo_vecinos_no_probabilidad': round(res.apoyo_vecinos, 5),
            'distancia_pca': round(res.distancia_vecino_mas_cercano, 5),
        })
        if args.graficas:
            graficas.append((matriz, ruta.name, res))
            destino = BASE / 'resultados' / ('resultado_' + ruta.stem + '.png')
            guardar_resultado(matriz, ruta.name, res, destino)

    metricas = None
    if evaluar:
        metricas = {
            'numero_muestras_independientes': len(verdad),
            'exactitud': float(accuracy_score(verdad, predicciones)),
            'etiquetas': sorted(set(verdad) | set(predicciones)),
        }
        metricas['matriz_confusion'] = confusion_matrix(verdad, predicciones, labels=metricas['etiquetas']).tolist()
        print(f'\nEXACTITUD CON ESTAS {len(verdad)} IMÁGENES DE PRUEBA: {metricas["exactitud"]*100:.1f}%')
        print(classification_report(verdad, predicciones, zero_division=0))
        print('Advertencia: unas pocas fotografías NO garantizan que funcionará igual con nuevas personas o luces.')

    if args.graficas:
        carpeta = BASE / 'resultados'
        carpeta.mkdir(exist_ok=True)
        guardar_eigenfaces(modelo, carpeta / 'eigenfaces.png')
        guardar_resumen_pruebas(graficas, carpeta / 'resumen_reconocimientos.png')
        (carpeta / 'resultados.json').write_text(
            json.dumps({'pruebas': resultados, 'metricas': metricas}, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(f'Imágenes y resumen guardados aquí: {carpeta}')
    if evaluar and args.ventana:
        abrir_panel_visual(graficas, resultados, metricas)
    return 0



def abrir_panel_visual(graficas, resultados, metricas):
    """Abre el panel de resultados y la cámara local."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        import numpy as np
        import cv2
        from PIL import Image, ImageTk

        ventana = tk.Tk()
        ventana.title('PCA + KNN | Reconocimiento facial educativo')
        ventana.geometry('1220x900')
        ventana.minsize(820, 620)
        ventana.configure(bg='#101a2b')

        AZUL = '#2a84d4'
        FONDO = '#101a2b'
        TARJETA = '#1a2940'
        TEXTO = '#edf3fc'
        MUTED = '#b2c0d1'
        VERDE = '#4fe0a0'
        ROJO = '#ff8f8f'
        camara = {'cap': None, 'after': None, 'activo': False, 'foto': None,
                  'modo': 'deteccion', 'ultimo': 0.0, 'resultado': None}
        modelo_reales = {'modelo': None}
        import time
        fotos = []

        encabezado = tk.Frame(ventana, bg=FONDO, padx=24, pady=13)
        encabezado.pack(fill='x')
        tk.Label(encabezado, text='RECONOCIMIENTO FACIAL | PCA + KNN',
                 font=('Segoe UI', 18, 'bold'), bg=FONDO, fg=TEXTO).pack(anchor='w')
        tk.Label(encabezado, text='Detección con cámara y coincidencias experimentales con imágenes locales autorizadas.',
                 font=('Segoe UI', 10), bg=FONDO, fg=MUTED).pack(anchor='w', pady=(3, 0))

        barra = tk.Frame(ventana, bg=FONDO, padx=24)
        barra.pack(fill='x', pady=(0, 12))
        if resultados:
            tk.Label(barra, text=f'PRUEBAS LOCALES: {len(resultados)}', bg=TARJETA, fg=TEXTO,
                     font=('Segoe UI', 11, 'bold'), padx=14, pady=10).pack(side='left', padx=(0, 9))
            aciertos = sum(r['prediccion'] == r['etiqueta_esperada'] for r in resultados)
            tk.Label(barra, text=f'ACIERTOS: {aciertos}/{len(resultados)}', bg=TARJETA,
                     fg=VERDE, font=('Segoe UI', 11, 'bold'), padx=14, pady=10).pack(side='left', padx=(0, 9))
            tk.Label(barra, text='Resultados experimentales, no verificación de identidad', bg=TARJETA,
                     fg=MUTED, font=('Segoe UI', 10), padx=12, pady=10).pack(side='left')
        else:
            tk.Label(barra, text='SIN IMÁGENES DE ENTRENAMIENTO INCLUIDAS', bg=TARJETA, fg=MUTED,
                     font=('Segoe UI', 11, 'bold'), padx=14, pady=10).pack(side='left')

        tabs = tk.Frame(ventana, bg=FONDO, padx=24)
        tabs.pack(fill='x')
        area = tk.Frame(ventana, bg=FONDO)
        area.pack(fill='both', expand=True)
        resultados_frame = tk.Frame(area, bg=FONDO)
        camara_frame = tk.Frame(area, bg=FONDO)
        estado = tk.StringVar(value='La cámara está apagada. No se guardan imágenes ni video.')
        titulo_camara = tk.StringVar(value='Detección de rostros en vivo')
        vista_camara = tk.Label(camara_frame, text='Cámara desactivada', bg=TARJETA, fg=MUTED,
                                font=('Segoe UI', 14), anchor='center')
        boton_camara = {'widget': None}

        def detener_camara():
            camara['activo'] = False
            camara['resultado'] = None
            if camara['after'] is not None:
                try:
                    ventana.after_cancel(camara['after'])
                except (ValueError, tk.TclError):
                    pass
                camara['after'] = None
            if camara['cap'] is not None:
                camara['cap'].release()
                camara['cap'] = None
            vista_camara.configure(image='', text='Cámara desactivada')
            camara['foto'] = None
            estado.set('La cámara está apagada. No se guardan imágenes ni video.')
            if boton_camara['widget'] is not None:
                boton_camara['widget'].configure(text='Iniciar cámara')

        def actualizar_camara(detector):
            camara['after'] = None
            if not camara['activo'] or camara['cap'] is None:
                return
            ok, frame = camara['cap'].read()
            if not ok:
                estado.set('No se pudo leer un fotograma. Comprueba los permisos de cámara.')
                detener_camara()
                return
            frame = cv2.flip(frame, 1)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostros = detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))
            modo_identidad = camara['modo'] == 'reconocimiento' and modelo_reales['modelo'] is not None
            etiqueta = None
            if len(rostros) and modo_identidad:
                rostro = max(rostros, key=lambda r: int(r[2]) * int(r[3]))
                ahora = time.monotonic()
                if ahora - camara['ultimo'] >= 0.35:
                    roi = recortar_rostro(gris, rostro)
                    _, vector = preparar_rostro_desde_gris(roi, modelo_reales['modelo'].tamano_imagen)
                    camara['resultado'] = modelo_reales['modelo'].predecir(vector)
                    camara['ultimo'] = ahora
                if camara['resultado'] is not None:
                    etiqueta = camara['resultado'].etiqueta
            elif not len(rostros):
                camara['resultado'] = None
            for x, y, w, h in rostros:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (65, 220, 150), 3)
            if modo_identidad:
                if not len(rostros):
                    estado.set('No se detecta ningún rostro. Prueba mejor iluminación y mira de frente a la cámara.')
                elif etiqueta:
                    estado.set(f'Coincidencia tentativa: {etiqueta}. No es verificación de identidad.')
                    x, y, w, h = max(rostros, key=lambda r: int(r[2]) * int(r[3]))
                    # OpenCV no dibuja Unicode correctamente; el nombre se muestra abajo.
                    cv2.putText(frame, 'Prediccion tentativa', (x, max(24, y - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 230, 170), 2, cv2.LINE_AA)
                else:
                    estado.set('Rostro detectado; evaluando coincidencias...')
            else:
                estado.set(f'Rostros detectados: {len(rostros)}. Modo detección, no identificación.'
                           if len(rostros) else
                           'Rostros detectados: 0. Aumenta la iluminación y mira de frente a la cámara.')
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Escala la imagen manteniendo la proporción del fotograma.
            ancho_disponible = max(320, vista_camara.winfo_width() - 12)
            alto_disponible = max(230, vista_camara.winfo_height() - 12)
            factor = min(ancho_disponible / img.width, alto_disponible / img.height)
            destino = (max(1, int(img.width * factor)), max(1, int(img.height * factor)))
            img = img.resize(destino, Image.Resampling.BILINEAR)
            foto = ImageTk.PhotoImage(img)
            camara['foto'] = foto
            vista_camara.configure(image=foto, text='')
            camara['after'] = ventana.after(40, lambda: actualizar_camara(detector))

        def alternar_camara():
            if camara['activo']:
                detener_camara()
                return
            try:
                detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if detector.empty():
                    raise RuntimeError('No se encontró el detector de rostros de OpenCV.')
                # Prueba varios backends por compatibilidad con webcams de Windows.
                intentos = ([(0, cv2.CAP_DSHOW), (0, cv2.CAP_ANY),
                             (1, cv2.CAP_DSHOW), (1, cv2.CAP_ANY)]
                            if sys.platform.startswith('win')
                            else [(0, cv2.CAP_ANY), (1, cv2.CAP_ANY)])
                cap = None
                for indice, backend in intentos:
                    prueba = cv2.VideoCapture(indice, backend)
                    if prueba.isOpened():
                        ok, imagen = prueba.read()
                        if ok and imagen is not None:
                            cap = prueba
                            break
                    prueba.release()
                if cap is None:
                    raise RuntimeError('No se pudo abrir una cámara disponible. Cierra otras apps que la usen y revisa los permisos de Windows.')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                camara['cap'] = cap
                camara['activo'] = True
                camara['ultimo'] = 0.0
                camara['resultado'] = None
                boton_camara['widget'].configure(text='Detener cámara')
                actualizar_camara(detector)
            except Exception as exc:
                detener_camara()
                messagebox.showerror('Error de cámara', str(exc), parent=ventana)

        def cargar_modelo_local():
            """Entrena exclusivamente con fotos locales autorizadas; no las sube."""
            raiz = DATOS_REALES
            revision = auditar_datos(raiz / 'personas', raiz / 'prueba')
            if not revision.valido:
                messagebox.showwarning(
                    'Faltan fotografías válidas',
                    'Para probar reconocimiento, guarda al menos 3 fotos DIFERENTES de cada una de 2 personas '
                    'autorizadas en datos/personas/NOMBRE/. Se recomiendan 8-15 por persona.\n\n'
                    + revision.mostrar(), parent=ventana)
                return False
            try:
                ventana.configure(cursor='watch')
                ventana.update_idletasks()
                X, y, _ = cargar_dataset(raiz / 'personas', (80, 80), detectar_rostros=True)
                m = ReconocedorEigenfaces((80, 80), componentes=12, vecinos=3)
                m.entrenar(X, y)
                modelo_reales['modelo'] = m
                if m.umbral_desconocido is None:
                    messagebox.showwarning(
                        'Reconocimiento experimental',
                        'El modelo no pudo calibrar el rechazo de personas desconocidas. '
                        'Puede proponer una coincidencia equivocada. Nunca uses esto para verificar identidades.',
                        parent=ventana)
                return True
            except Exception as exc:
                messagebox.showerror('No se pudo entrenar', str(exc), parent=ventana)
                return False
            finally:
                ventana.configure(cursor='')

        def cambiar_tab(modo):
            if modo == 'reconocimiento' and modelo_reales['modelo'] is None:
                if not cargar_modelo_local():
                    return
            detener_camara()
            resultados_frame.pack_forget()
            camara_frame.pack_forget()
            if modo == 'resultados':
                resultados_frame.pack(fill='both', expand=True)
            else:
                camara['modo'] = 'reconocimiento' if modo == 'reconocimiento' else 'deteccion'
                titulo_camara.set('Coincidencias tentativas con fotos locales (no verifica identidades)'
                                  if modo == 'reconocimiento' else 'Detección de rostros en vivo')
                camara_frame.pack(fill='both', expand=True)
                ventana.after(150, lambda: alternar_camara() if camara_frame.winfo_ismapped() and not camara['activo'] else None)

        def boton(parent, texto, comando):
            return tk.Button(parent, text=texto, command=comando, font=('Segoe UI', 10, 'bold'),
                             bg=AZUL, fg='white', activebackground='#1664aa', activeforeground='white',
                             relief='flat', padx=14, pady=9, cursor='hand2')

        boton(tabs, 'Ver resultados', lambda: cambiar_tab('resultados')).pack(side='left', padx=(0, 10))
        boton(tabs, 'Probar cámara (detección)', lambda: cambiar_tab('camara')).pack(side='left', padx=(0, 10))
        boton(tabs, 'Reconocer (fotos locales)', lambda: cambiar_tab('reconocimiento')).pack(side='left')

        def alternar_pantalla_completa(event=None):
            ventana.attributes('-fullscreen', not bool(ventana.attributes('-fullscreen')))

        ventana.bind('<F11>', alternar_pantalla_completa)
        ventana.bind('<Escape>', lambda event: ventana.attributes('-fullscreen', False))
        boton(tabs, 'Ampliar (F11)', alternar_pantalla_completa).pack(side='right')

        canvas = tk.Canvas(resultados_frame, bg=FONDO, highlightthickness=0)
        scroll = tk.Scrollbar(resultados_frame, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        rejilla = tk.Frame(canvas, bg=FONDO)
        interior = canvas.create_window((0, 0), window=rejilla, anchor='nw')
        rejilla.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(interior, width=e.width))

        for i, (matriz, nombre, _) in enumerate(graficas):
            dato = resultados[i]
            correcto = dato['prediccion'] == dato['etiqueta_esperada']
            tarjeta = tk.Frame(rejilla, bg=TARJETA, padx=11, pady=11)
            tarjeta.grid(row=i // 3, column=i % 3, sticky='nsew', padx=10, pady=9)
            rejilla.grid_columnconfigure(i % 3, weight=1, uniform='tarjetas')
            imagen = Image.fromarray(np.uint8(np.clip(matriz * 255, 0, 255)), 'L').resize((160, 160), Image.Resampling.NEAREST)
            foto = ImageTk.PhotoImage(imagen)
            fotos.append(foto)
            tk.Label(tarjeta, image=foto, bg=TARJETA).pack()
            tk.Label(tarjeta, text=f"Predicción: {dato['prediccion']}", bg=TARJETA,
                     fg=VERDE if correcto else ROJO, font=('Segoe UI', 11, 'bold')).pack(pady=(9, 2))
            tk.Label(tarjeta, text=f"Esperado: {dato['etiqueta_esperada']}", bg=TARJETA,
                     fg=MUTED, font=('Segoe UI', 9)).pack()
            tk.Label(tarjeta, text=nombre, bg=TARJETA, fg=MUTED, font=('Segoe UI', 8)).pack(pady=(3,0))

        if not resultados:
            guia = tk.Frame(rejilla, bg=TARJETA, padx=24, pady=22)
            guia.grid(row=0, column=0, padx=24, pady=24, sticky='ew')
            tk.Label(guia, text='Añade tus propias imágenes de ejemplo para entrenar',
                     bg=TARJETA, fg=TEXTO, font=('Segoe UI', 15, 'bold')).pack(anchor='w')
            tk.Label(guia, justify='left', bg=TARJETA, fg=MUTED, font=('Segoe UI', 11),
                     text='1. Guarda fotos autorizadas y diferentes en datos/personas/Persona_A/ y Persona_B/.\n'
                          '2. Coloca OTRAS fotos diferentes en datos/prueba/Persona_A/ y Persona_B/.\n'
                          '3. Cierra y vuelve a pulsar ▶ Ejecutar para evaluar las imágenes.\n\n'
                          'Mínimo: 3 fotos distintas por persona; recomendado: 8-15.\n'
                          'Puedes probar la detección de la cámara sin añadir imágenes.\n'
                          'No subas fotos de personas ni resultados privados a GitHub.'
                     ).pack(anchor='w', pady=(12, 0))

        # El modo detección funciona sin entrenamiento; las coincidencias sí lo requieren.
        tk.Label(camara_frame, textvariable=titulo_camara,
                 font=('Segoe UI', 13, 'bold'), bg=FONDO, fg=TEXTO).pack(pady=(10, 6))
        boton_camara['widget'] = boton(camara_frame, 'Iniciar cámara', alternar_camara)
        boton_camara['widget'].pack(pady=(0, 5))
        vista_camara.pack(padx=15, pady=3, fill='both', expand=True)
        tk.Label(camara_frame, textvariable=estado, font=('Segoe UI', 10),
                 bg=FONDO, fg=MUTED).pack(pady=(4, 6))

        pie = tk.Frame(ventana, bg=FONDO, padx=24, pady=7)
        pie.pack(fill='x')

        def ver_eigenfaces():
            ruta = BASE / 'resultados' / 'eigenfaces.png'
            if not ruta.is_file():
                messagebox.showinfo('Eigenfaces', 'Todavía no se han generado las gráficas.', parent=ventana)
                return
            popup = tk.Toplevel(ventana)
            popup.title('PCA | Eigenfaces del entrenamiento local')
            popup.configure(bg=FONDO)
            img = Image.open(ruta).convert('RGB')
            img.thumbnail((900, 650), Image.Resampling.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            etiqueta = tk.Label(popup, image=foto, bg=FONDO)
            etiqueta.image = foto
            etiqueta.pack(padx=10, pady=10)

        boton(pie, 'Ver Eigenfaces', ver_eigenfaces).pack(side='left')
        tk.Label(pie, text='Las fotografías y los resultados permanecen en tu equipo.',
                 bg=FONDO, fg=MUTED, font=('Segoe UI', 9)).pack(side='right')

        cambiar_tab('resultados')
        def cerrar():
            detener_camara()
            ventana.destroy()
        ventana.protocol('WM_DELETE_WINDOW', cerrar)
        print('Ventana abierta. Usa las pestañas de detección o reconocimiento experimental.')
        ventana.mainloop()
    except (ImportError, OSError) as exc:
        print(f'No se pudo abrir la ventana visual: {exc}')
        print('Puedes revisar resultados/resumen_reconocimientos.png con el explorador de archivos.')
    except Exception as exc:
        print(f'No fue posible abrir la ventana gráfica: {type(exc).__name__}: {exc}')
        print('Revisa los archivos en la carpeta resultados/.')


def ejecutar(args):
    if any(n <= 0 for n in args.tamano):
        raise ValueError('El ancho y el alto deben ser positivos.')
    if args.imagen:
        args.modo = 'imagenes'
    if args.modo == 'diagnostico':
        return 0 if auditar(args).valido else 2
    if args.modo == 'menu':
        print('=== EIGENFACES + KNN ===')
        print('1. Auditar fotografías\n2. Reconocer imágenes\n3. Evaluar con fotos independientes\n4. Cámara\n0. Salir')
        opciones = {'1': 'diagnostico', '2': 'imagenes', '3': 'evaluar', '4': 'camara', '0': 'salir'}
        args.modo = opciones.get(input('Elige una opción: ').strip(), 'salir')
        if args.modo == 'salir':
            return 0
        if args.modo == 'diagnostico':
            return 0 if auditar(args).valido else 2

    if args.ventana and args.modo == 'evaluar':
        raiz = carpeta_datos(args)
        revision = auditar_datos(raiz / 'personas', raiz / 'prueba')
        if not revision.valido or not listar_imagenes(raiz / 'prueba'):
            print('No hay fotografías suficientes para evaluar. Se abre el panel de instrucciones.')
            print('Añade imágenes locales autorizadas en datos/personas/ y datos/prueba/.')
            abrir_panel_visual([], [], None)
            return 0

    modelo = entrenar(args)
    if args.modo in ('imagenes', 'evaluar'):
        if args.modo == 'evaluar' and args.imagen:
            raise ValueError('Para evaluar no uses --imagen: separa fotos de prueba por carpeta.')
        return ejecutar_imagenes(args, modelo, evaluar=(args.modo == 'evaluar'))
    if args.modo == 'camara':
        return ejecutar_camara(modelo, args.camara, args.umbral)
    raise ValueError(f'Modo no soportado: {args.modo}')

if __name__ == '__main__':
    try:
        sys.exit(ejecutar(parsear_argumentos()))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'\n[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
