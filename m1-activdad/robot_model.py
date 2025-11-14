# robot_model.py
# Modelo de robots de limpieza reactivos usando Mesa

import numpy as np
import pandas as pd
import seaborn as sns

import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle


# ==========================
# MÉTRICAS / FUNCIONES AUXILIARES
# ==========================

def compute_clean_percentage(model: "CleaningModel") -> float:
    """Porcentaje de celdas limpias en el modelo."""
    total_cells = model.width * model.height
    dirty_cells = len(model.dirty_cells)
    clean_cells = total_cells - dirty_cells
    if total_cells == 0:
        return 0.0
    return 100.0 * clean_cells / total_cells


def compute_total_moves(model: "CleaningModel") -> int:
    """Número total de movimientos realizados por todos los agentes."""
    return sum(agent.moves for agent in model.agents)


# ==========================
# AGENTE (ROBOT DE LIMPIEZA)
# ==========================

class CleaningAgent(CellAgent):
    """Robot de limpieza reactivo.

    Reglas:
    - Si la celda actual está sucia: aspira (la limpia).
    - Si la celda actual está limpia: se mueve a una celda vecina aleatoria.
    """

    def __init__(self, model: "CleaningModel", cell):
        """Crea un nuevo agente de limpieza.

        Args:
            model: instancia del modelo
            cell: celda inicial donde empieza el agente
        """
        super().__init__(model)
        self.cell = cell
        self.moves = 0  # movimientos efectivos (cambio de celda)

    def _is_current_cell_dirty(self) -> bool:
        """Regresa True si la celda actual está sucia."""
        coord = self.cell.coordinate
        return coord in self.model.dirty_cells

    def clean(self) -> None:
        """Limpia la celda actual si está sucia."""
        coord = self.cell.coordinate
        if coord in self.model.dirty_cells:
            self.model.dirty_cells.remove(coord)

    def move(self) -> None:
        """Se mueve a una celda vecina aleatoria (vecindad de Moore, 8 direcciones)."""
        old_cell = self.cell
        new_cell = self.cell.neighborhood.select_random_cell()
        self.cell = new_cell

        # Contamos movimiento solo si cambió de celda
        if self.cell is not old_cell:
            self.moves += 1

    def step(self) -> None:
        """Un paso de comportamiento del agente."""
        if self._is_current_cell_dirty():
            # Si la celda está sucia, aspira
            self.clean()
        else:
            # Si está limpia, se mueve aleatoriamente
            self.move()


# ==========================
# MODELO
# ==========================

class CleaningModel(mesa.Model):
    """Modelo de robots de limpieza sobre una grilla MxN.

    Parámetros:
        n_agents: Número de robots.
        width: Ancho (M) de la habitación.
        height: Alto (N) de la habitación.
        dirty_percentage: Porcentaje inicial de celdas sucias (0.0 a 1.0).
        max_steps: Tiempo máximo (número de pasos de simulación).
    """

    def __init__(
        self,
        n_agents: int = 5,
        width: int = 10,
        height: int = 10,
        dirty_percentage: float = 0.3,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)

        self.num_agents = n_agents
        self.width = width
        self.height = height
        self.dirty_percentage = dirty_percentage
        self.max_steps = max_steps

        # Control de ejecución
        self.running = True
        self.current_step = 0  # tiempo de simulación (steps)

        # Grid con vecindad de Moore (8 vecinos)
        self.grid = OrthogonalMooreGrid((width, height), random=self.random)

        # ---------- Inicializar celdas sucias ----------
        all_cells = list(self.grid.all_cells.cells)
        total_cells = width * height
        n_dirty = int(self.dirty_percentage * total_cells)
        n_dirty = max(0, min(n_dirty, total_cells))

        dirty_sample = self.random.sample(all_cells, k=n_dirty)
        # Conjunto de coordenadas sucias
        self.dirty_cells: set[tuple[int, int]] = {
            cell.coordinate for cell in dirty_sample
        }

        # ---------- Crear agentes ----------
        # Todos empiezan en la celda [1,1] → coordenada (0,0) en código
        start_cell = self.grid[(0, 0)]

        CleaningAgent.create_agents(
            self,
            self.num_agents,
            [start_cell] * self.num_agents,
        )

        # ---------- DataCollector ----------
        # Aquí recopilamos durante la ejecución:
        # - Step: tiempo actual
        # - CleanPercentage: porcentaje de celdas limpias
        # - TotalMoves: movimientos totales de todos los agentes
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": lambda m: m.current_step,
                "CleanPercentage": compute_clean_percentage,
                "TotalMoves": compute_total_moves,
            },
            agent_reporters={
                "Moves": "moves",
            },
        )

        # Registrar estado inicial
        self.datacollector.collect(self)

    # ---------- Utilidades del modelo ----------

    def all_clean(self) -> bool:
        """True si ya no hay celdas sucias."""
        return len(self.dirty_cells) == 0

    # ---------- Paso de simulación ----------

    def step(self) -> None:
        """Un paso del modelo."""
        # Revisamos condición de paro ANTES de avanzar
        if self.all_clean() or self.current_step >= self.max_steps:
            self.running = False
            return

        self.current_step += 1

        # Todos los agentes actúan en orden aleatorio
        self.agents.shuffle_do("step")

        # Registrar datos
        self.datacollector.collect(self)

        # Revisamos nuevamente condición de paro
        if self.all_clean() or self.current_step >= self.max_steps:
            self.running = False


# ==========================
# VISUALIZACIÓN (GRID + GRÁFICAS EN OTRA PÁGINA)
# ==========================

def agent_portrayal(agent: CleaningAgent) -> AgentPortrayalStyle:
    """Cómo se dibuja cada agente en el grid.

    Rojo si la celda donde está es sucia.
    Verde si está limpia.
    """
    coord = agent.cell.coordinate
    is_dirty = coord in agent.model.dirty_cells
    color = "tab:red" if is_dirty else "tab:green"
    return AgentPortrayalStyle(color=color, size=50)


# Parámetros del modelo para la UI de Mesa/Solara
model_params = {
    "n_agents": {
        "type": "SliderInt",
        "value": 5,
        "label": "Número de agentes:",
        "min": 1,
        "max": 50,
        "step": 1,
    },
    "width": {
        "type": "SliderInt",
        "value": 10,
        "label": "Ancho (M):",
        "min": 5,
        "max": 30,
        "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": 10,
        "label": "Alto (N):",
        "min": 5,
        "max": 30,
        "step": 1,
    },
    "dirty_percentage": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Porcentaje inicial de celdas sucias:",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "max_steps": {
        "type": "SliderInt",
        "value": 200,
        "label": "Tiempo máximo (steps):",
        "min": 10,
        "max": 1000,
        "step": 10,
    },
}

# Página 1: gráficas (vs Step)
CleanPlot = make_plot_component("CleanPercentage", page=1)
MovesPlot = make_plot_component("TotalMoves", page=1)

# Modelo inicial para la visualización interactiva
clean_model = CleaningModel(
    n_agents=5,
    width=10,
    height=10,
    dirty_percentage=0.3,
    max_steps=200,
)

# Render del espacio (Página 0)
renderer = SpaceRenderer(model=clean_model, backend="matplotlib").render(
    agent_portrayal=agent_portrayal
)

# SolaraViz:
# - Página 0: grid del modelo
# - Página 1: gráficas CleanPercentage y TotalMoves
page = SolaraViz(
    clean_model,
    renderer,
    components=[CleanPlot, MovesPlot],
    model_params=model_params,
    name="Reactive Cleaning Robot",
)


# ==========================
# EXPERIMENTOS PARA EL REPORTE
# ==========================

def run_experiment(
    n_agents: int,
    width: int,
    height: int,
    dirty_percentage: float,
    max_steps: int,
    seed: int | None = None,
):
    """Ejecuta una simulación y regresa las 3 métricas finales.

    Regresa:
        time_used: tiempo (steps) hasta limpiar todo o llegar al máximo.
        final_clean_pct: porcentaje de celdas limpias al final.
        final_total_moves: número total de movimientos de todos los agentes.
        model: instancia final del modelo.
    """
    model = CleaningModel(
        n_agents=n_agents,
        width=width,
        height=height,
        dirty_percentage=dirty_percentage,
        max_steps=max_steps,
        seed=seed,
    )

    # Ejecutar hasta que termine (todo limpio o max_steps)
    while model.running:
        model.step()

    # 1) Tiempo necesario
    time_used = model.current_step

    # 2) Porcentaje de celdas limpias al final
    final_clean_pct = compute_clean_percentage(model)

    # 3) Número de movimientos de todos los agentes
    final_total_moves = compute_total_moves(model)

    return time_used, final_clean_pct, final_total_moves, model


# ==========================
# MAIN: correr experimentos y gráficas vs número de agentes
# ==========================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Cantidades de agentes a probar
    agent_values = [1, 2, 3, 5, 8, 10, 15, 20]

    rows = []

    for n in agent_values:
        time_used, final_clean_pct, final_total_moves, _ = run_experiment(
            n_agents=n,
            width=20,
            height=20,
            dirty_percentage=0.3,
            max_steps=1000,
            seed=123,
        )
        rows.append({
            "n_agents": n,
            "time_used": time_used,
            "final_clean_pct": final_clean_pct,
            "final_total_moves": final_total_moves,
        })

    df_results = pd.DataFrame(rows)
    print(df_results)

    # ========= Gráfica 1 — Tiempo vs número de agentes =========
    plt.figure(figsize=(7, 4))
    plt.plot(
        df_results["n_agents"],
        df_results["time_used"],
        marker="o",
        linestyle="-"
    )
    plt.title("Tiempo de limpieza vs número de agentes")
    plt.xlabel("Número de agentes")
    plt.ylabel("Tiempo (steps) hasta limpiar o llegar al máximo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========= Gráfica 2 — Movimientos totales vs número de agentes =========
    plt.figure(figsize=(7, 4))
    plt.plot(
        df_results["n_agents"],
        df_results["final_total_moves"],
        marker="o",
        linestyle="-"
    )
    plt.title("Movimientos totales vs número de agentes")
    plt.xlabel("Número de agentes")
    plt.ylabel("Número total de movimientos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
