# Robot de limpieza reactivo con Mesa y Solara

Este proyecto implementa un **sistema multiagente** de robots de limpieza usando la librería **Mesa** en Python, siguiendo la especificación de la actividad de clase:

- Habitación de `M x N` espacios (grid).
- Varios agentes (robots de limpieza).
- Porcentaje de celdas inicialmente sucias.
- Tiempo máximo de ejecución.

Cada robot se comporta de forma **reactiva**:

1. Todos los agentes comienzan en la celda `[1,1]` (coordenada `(0, 0)` en el código).
2. En cada paso de tiempo:
   - Si la celda está sucia → **aspira** (la limpia).
   - Si la celda está limpia → elige una dirección aleatoria (8 vecinos tipo Moore); si no puede moverse, se queda en la misma celda.

Durante la simulación se recopila:

- **Tiempo necesario** hasta que todas las celdas estén limpias (o se alcance el tiempo máximo).
- **Porcentaje de celdas limpias** al final de la simulación.
- **Número total de movimientos** realizados por todos los agentes.

Además, se incluye:

- Una **visualización interactiva** con Solara (grid + gráficas).
- Un modo de **experimentos** para analizar cómo impacta el número de agentes en el tiempo y en los movimientos.

---

## Estructura del proyecto

```text
.
├── robot_model.py   # Modelo, agentes, visualización y experimentos
├── README.md        # Este archivo
└── (opcional) data/ # Carpeta para guardar imágenes de gráficas, si se desea
