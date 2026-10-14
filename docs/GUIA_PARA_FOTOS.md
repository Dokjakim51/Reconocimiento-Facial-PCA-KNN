# Recolección de fotografías para evaluar reconocimiento facial

El repositorio no distribuye fotografías de entrenamiento ni de prueba. Cada persona que ejecute el proyecto debe preparar sus propias imágenes autorizadas y mantenerlas fuera de GitHub.

## Para cada persona participante

- Pedir permiso para fotografiarla y utilizar sus imágenes **localmente** en un ejercicio académico. No hace falta publicar esas fotos.
- Tomar unas 8 a 15 fotografías de entrenamiento **distintas**: frontal, un poco girado hacia ambos lados, luz natural e interior.
- Tomar otras 3 a 5 fotos nuevas para prueba. No deben ser fotos repetidas ni versiones del mismo archivo de entrenamiento.
- Organizar las imágenes en carpetas separadas por participante. Usa alias como `Persona_A` o `Persona_B`.
- Cuanto más variadas sean las fotos, mejor podrás estudiar los errores de la aplicación. No hay garantía de resultados, especialmente si los rostros son pequeños u ocultos.

## Carpetas

```text
datos/personas/Persona_A/entrenamiento1.jpg
datos/personas/Persona_A/entrenamiento2.jpg
datos/personas/Persona_B/entrenamiento1.jpg
datos/prueba/Persona_A/prueba_nueva1.jpg
datos/prueba/Persona_B/prueba_nueva1.jpg
```

La evaluación informa la exactitud **solo de esas imágenes independientes**; añade casos de prueba nuevos y analiza los errores. Cuando ya no necesites las fotografías de terceros, bórralas del proyecto local según lo que acordaste con ellos.

No publiques en GitHub rostros o datos biométricos de otras personas sin su consentimiento específico. Tampoco uses estos resultados para tomar decisiones reales de identidad o acceso.
