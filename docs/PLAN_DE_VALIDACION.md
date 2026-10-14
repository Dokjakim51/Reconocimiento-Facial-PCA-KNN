# Plan de validación del proyecto

## Comprobaciones automáticas

- Duplicados en entrenamiento.
- Detección de imágenes compartidas entre entrenamiento y prueba.
- Entrenamiento, predicción y rechazo orientativo sobre matrices sintéticas.
- Tamaño y consistencia del preprocesamiento.
- Primera ejecución sin fotografías: abre instrucciones en lugar de fallar.

Ejecutar con `python -m unittest discover -s tests -v`.

## Pruebas manuales pendientes con fotografías autorizadas

1. Usar al menos dos personas con fotos distintas de entrenamiento y de prueba.
2. Comprobar que la auditoría indica cero duplicados y cero filtraciones entre conjuntos.
3. Ejecutar la evaluación y registrar aciertos **y errores**, junto al número de muestras.
4. Verificar webcam, iluminación, detección frontal y cierre de la cámara.
5. Evaluar qué pasa con participantes no entrenados, sin confiar ciegamente en el umbral de desconocidos.
6. Confirmar que Git no incluye fotografías, videos ni resultados privados.

Las métricas reales aún no están verificadas; las pruebas artificiales no demuestran eficacia con rostros de personas. Este sistema es **educativo**.
