```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Feb__8_05:53:42_Coordinated_Universal_Time_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```

```bash
Python 3.10.8
```

# VERSION 1

## Tecnica

- Descenso de Gradiente
- Inicializacion aleatoria
- Las posiciones (x, y, z), colores (r, g, b) y tamaños de las gaussianas se generan de forma aleatoria.
- Cada gaussiana se representa como un círculo isotrópico (sin rotación / normal) en la imagen.
- Se aplica una máscara gaussiana
- Optimización de Adam: Se ajustan las posiciones, los colores y los tamaños de las gaussianas para aproximar la imagen objetivo.
- Gradient Clipping: Se limitan los gradientes para evitar que el entrenamiento sea inestable.

## Resultado

- La imagen generada tenía un aspecto borroso y con manchas dispersas.
- Las gaussianas eran isotrópicas (círculos puros), lo cual hacía difícil ajustarse a formas complejas y detalles finos.
- Limitación: No había rotación ni adaptación de forma, lo cual es un problema para capturar bordes.

# VERSION 2

## Tecnica

- Decaimiento Progresivo del Tamaño:
    - A medida que avanza el entrenamiento, el tamaño de las gaussianas se reduce progresivamente
    - Permite que las gaussianas se afinen y se adapten mejor a los detalles pequeños de la imagen.
- Scheduler del Learning Rate:
    - Se aplica un scheduler para reducir el learning rate cada 150 iteraciones, ayudando a la estabilidad del ajuste.

## Resultado

- Hubo una mejora visible respecto a la versión 1: la imagen era más estable y definida.
- Problema: Aún se veían zonas borrosas.
- Limitación: Las gaussianas seguían siendo círculos sin rotación, y las áreas de bordes seguían sin representarse bien.

# VERSION 3

## Tecnica

- Gaussianas Elípticas y Rotadas
- Inicialización en espacio tanh: Las posiciones y colores se inicializan en un espacio tanh, que permite una mayor exploración en zonas complicadas.
- Alpha Blending:
    - Cada gaussiana incluye un valor de transparencia (alpha) que se optimiza.
    - Esto ayuda a superponer gaussianas sin saturar el color.

## Resultado

- Las zonas de la imagen se ajustaban de manera más suave y precisa.
- Mejora clave: El hecho de que las gaussianas puedan rotar y adaptarse elípticamente mejora mucho la calidad.

# VERSION 4

## Tecnica

- Construcción de un Kernel 2D con Rotación y Escala:
- Transformación afín para colocar los splats en el plano 2D:
    - Una vez generados los kernels, se traducen al plano de imagen usando una transformación afín.
    - Esto se realiza con grid_sample, que permite una proyección precisa en la imagen.
- Re-muestreo adaptativo (Densificación):
    - Esta técnica añade o clona gaussianas en puntos donde el gradiente es grande o la proyección es insuficiente.
    - Se identifican regiones de alto error y se ajustan dinámicamente.
- Uso de SSIM Loss para optimización:
    - Además de la pérdida L1 clásica, se utiliza el índice de similitud estructural (SSIM) para mejorar la percepción visual.

## Resultado

- Adaptación dinámica: Si un splat está mal posicionado o no contribuye, se reemplaza o se clona para cubrir mejor el espacio.
- Optimización perceptual (SSIM): No solo intenta parecerse en píxeles, sino que respeta la estructura visual (Structural Similarity Index).
- Transformación afín en lugar de proyección manual: El método que usan con affine_grid y grid_sample permite una precisión mucho mayor en la colocación del splat en la imagen.
