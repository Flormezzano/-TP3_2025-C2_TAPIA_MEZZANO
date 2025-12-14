# TP3 - Análisis de Tiradas de Dados

Este proyecto analiza videos de tiradas de dados para detectar automáticamente los momentos en que los dados están en reposo, identificar cada dado y contar el número de pips (puntos) en sus caras.

## Descripción

El script `TP3.py` procesa una serie de videos (`tirada_1.mp4`, `tirada_2.mp4`, etc.). Para cada video, realiza las siguientes acciones:

1.  **Detección de Movimiento**: Analiza el video para encontrar los segmentos donde no hay movimiento.
2.  **Selección de Frame**: Elige el frame más representativo del primer período de reposo detectado.
3.  **Detección de Dados**: En el frame seleccionado, localiza los 5 dados presentes en la escena.
4.  **Conteo de Pips**: Para cada dado detectado, cuenta el número de pips en su cara visible.
5.  **Generación de Resultados**: Guarda imágenes de depuración (los dados recortados, el frame con los dados marcados, etc.) en la carpeta `debug_reposo/tirada_<i>`.
6.  **Video Anotado**: Crea un nuevo video (`tirada_<i>_anotada.mp4`) que muestra los rectángulos delimitadores y el valor de cada dado durante los períodos de reposo.

## Requisitos

Para ejecutar este script, necesitás tener instaladas las siguientes bibliotecas de Python:

-   `opencv-python`
-   `numpy`
-   `matplotlib`

Podés instalarlas usando pip:
```bash
pip install opencv-python numpy matplotlib
```

## Uso

Para ejecutar el análisis, simplemente ejecutá el script `TP3.py` desde la terminal. Asegúrate de que los videos de las tiradas (`tirada_1.mp4`, `tirada_2.mp4`, etc.) se encuentren en el mismo directorio que el script.

```bash
python TP3.py
```

El script procesará todos los videos `tirada_*.mp4` que encuentres en la carpeta.

## Estructura de Archivos

```
.
├── TP3.py                # Script principal de análisis
├── tirada_1.mp4          # Video de la primera tirada
├── tirada_2.mp4          # Video de la segunda tirada
├── ...
└── debug_reposo/         # Carpeta de salida para los resultados
    ├── tirada_1/
    │   ├── frame_reposo.png
    │   ├── dado_1.png
    │   ...
    │   └── tirada_1_anotada.mp4
    └── tirada_2/
        └── ...
```

## Salida del Script

El script generará lo siguiente:

-   **Salida en la consola**: Imprime información sobre los segmentos de reposo encontrados y el número de pips detectados para cada dado en cada tirada.
-   **Imágenes de depuración**: En la carpeta `debug_reposo/tirada_<i>/`, encontrarás imágenes que muestran el frame analizado, los dados detectados y los pips contados.
-   **Videos anotados**: En la misma carpeta de depuración, se guardará un video llamado `tirada_<i>_anotada.mp4` con los resultados visuales del análisis.
-   **Visualización con Matplotlib**: Si se ejecuta en un entorno con GUI, mostrará los resultados de cada tirada usando `matplotlib`.