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

## Cómo descargar este repositorio

Hay dos formas de obtener el proyecto en tu computadora.

### Opción 1: Clonar con Git (recomendado)

1. Abre una terminal.
2. Ve a la carpeta donde quieras guardar el proyecto, por ejemplo:

   ```bash
   cd ~/Desktop
   ```

3. Clona el repositorio (ajusta la URL si es diferente):

   ```bash
   git clone https://github.com/MultiagentesDuplas/Multiagentes-Actividades.git
   ```

4. Entra a la carpeta del repositorio:

   ```bash
   cd Multiagentes-Actividades
   ```

5. (Opcional) Si este proyecto está en una subcarpeta (por ejemplo `robot-limpieza/`), entra ahí:

   ```bash
   cd robot-limpieza
   ```

### Opción 2: Descargar como ZIP desde GitHub

1. Entra a la página del repositorio en GitHub.
2. Haz clic en el botón verde **Code**.
3. Selecciona **Download ZIP**.
4. Descomprime el ZIP en tu computadora.
5. Abre una terminal y ve a la carpeta donde descomprimiste el proyecto, por ejemplo:

   ```bash
   cd ~/Desktop/Multiagentes-Actividades
   ```

> A partir de aquí, las instrucciones de instalación y ejecución son iguales para ambas opciones.

---

## Requisitos

- Python 3.10+ (recomendado)
- `pip`

Librerías de Python usadas:

- `mesa`
- `solara`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

---

## Instalación del entorno (una sola vez)

Se recomienda usar un entorno virtual para no mezclar dependencias con otros proyectos.

### 1. Crear y activar entorno virtual

Desde la carpeta del proyecto:

**En Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**En Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Cada vez que quieras trabajar con el proyecto, primero activa el entorno virtual.

### 2. Instalar dependencias

Con el entorno virtual activado:

```bash
pip install mesa solara pandas numpy matplotlib seaborn
```

---

## Estructura del proyecto

Ejemplo de estructura mínima:

```text
.
├── robot_model.py   # Modelo, agentes, visualización y experimentos
├── README.md        # Este archivo
├── .gitignore       # Para ignorar .venv, __pycache__, etc.
└── (opcional) data/ # Carpeta para guardar imágenes de gráficas, si se desea
```

El archivo principal es `robot_model.py`.  
Ahí se definen:

- `CleaningAgent`: agente (robot de limpieza).
- `CleaningModel`: modelo de la habitación y las reglas.
- `page`: objeto de Solara para correr con `solara run`.
- `run_experiment(...)`: función para correr simulaciones y obtener métricas.
- Bloque `if __name__ == "__main__":` para generar gráficas de análisis.

---

## Cómo correr la visualización interactiva (Solara)

Este modo muestra:

- En la **Página 0**: la habitación con los robots moviéndose.
- En la **Página 1**: gráficas del porcentaje de celdas limpias y de los movimientos totales a lo largo del tiempo.

Con el entorno virtual activado, ejecuta:

```bash
solara run robot_model.py
```

Luego abre en el navegador la URL que te indique (por ejemplo `http://localhost:8765`).

En la interfaz podrás:

- Cambiar el **número de agentes** (`n_agents`).
- Cambiar el **tamaño de la habitación** (`width`, `height`).
- Modificar el **porcentaje inicial de celdas sucias** (`dirty_percentage`).
- Ajustar el **tiempo máximo** en pasos (`max_steps`).
- Ver la animación de los robots sobre la grilla.
- Ir a la pestaña/página de gráficas para ver cómo evoluciona:
  - `CleanPercentage` (% de celdas limpias).
  - `TotalMoves` (movimientos totales).

---

## Cómo correr los experimentos para el reporte

El archivo también incluye una función:

```python
run_experiment(n_agents, width, height, dirty_percentage, max_steps, seed=None)
```

que devuelve:

- `time_used`: tiempo (steps) hasta que se limpiaron todas las celdas o se alcanzó el máximo.
- `final_clean_pct`: porcentaje de celdas limpias al final.
- `final_total_moves`: número total de movimientos de todos los agentes.
- `model`: instancia final del modelo.

Además, en `robot_model.py` hay un bloque:

```python
if __name__ == "__main__":
    ...
```

que automáticamente:

1. Ejecuta varios experimentos variando el número de agentes.
2. Imprime una tabla (`DataFrame`) con:
   - `n_agents`
   - `time_used`
   - `final_clean_pct`
   - `final_total_moves`
3. Muestra dos gráficas:
   - **Figura 1:** Tiempo de limpieza vs número de agentes.
   - **Figura 2:** Movimientos totales vs número de agentes.

Para correr este modo (sin Solara, solo análisis):

```bash
python robot_model.py
```

> Resumen:  
> - `solara run robot_model.py` → interfaz interactiva en el navegador.  
> - `python robot_model.py` → corre experimentos y muestra gráficas con matplotlib.

---

## Interpretación de las métricas

Las métricas recopiladas permiten analizar:

- **Tiempo de limpieza (`time_used`)**  
  Cómo disminuye (o no) el tiempo de limpieza al aumentar el número de agentes.

- **Porcentaje final de limpieza (`final_clean_pct`)**  
  Qué tan efectiva fue la limpieza cuando la simulación se detuvo.

- **Movimientos totales (`final_total_moves`)**  
  Cuántos movimientos en total realizaron todos los robots para lograr la limpieza.

Estas métricas se pueden usar para:

- Comparar configuraciones con distintos números de agentes.
- Analizar si más agentes siempre ayudan o si hay un punto de rendimiento decreciente.
- Incluir tablas y gráficas en el informe en PDF que pide la actividad.

---

## Autoría

- Integrantes del equipo:
  - Yael Sinuhe Grajeda Martínez — A01801044  
  - Tadeo Emanuel Arellano Conde — A01800840
- Curso: Sistemas Multiagentes
- Institución: Tecnológico de Monterrey, Campus CEM