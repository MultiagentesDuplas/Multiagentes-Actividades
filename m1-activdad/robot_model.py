# robot_model.py
# Modelo de robots de limpieza reactivos usando Mesa.
# Autores: Arellano Conde, Tadeo Emanuel (A01800840)
#          Grajeda Martínez, Yael Sinuhe (A01801044)
# Fecha de última modificación: 14/nov/2025

import pandas as pd
import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle


# ==========================
# MÉTRICAS / FUNCIONES AUXILIARES
# ==========================

def computeCleanPercentage(model: "CleaningModel") -> float:
    """
    Calcula el porcentaje de celdas limpias en el modelo.

    Parámetros:
        model: instancia de CleaningModel.

    Regresa:
        Porcentaje de celdas limpias (0.0 a 100.0).
    """
    totalCells = model.width * model.height
    dirtyCells = len(model.dirtyCells)
    cleanCells = totalCells - dirtyCells
    if totalCells == 0:
        return 0.0
    return 100.0 * cleanCells / totalCells


def computeTotalMoves(model: "CleaningModel") -> int:
    """
    Calcula el número total de movimientos realizados por todos los agentes.

    Parámetros:
        model: instancia de CleaningModel.

    Regresa:
        Entero con el total de movimientos acumulados.
    """
    return sum(agent.moves for agent in model.agents)


# ==========================
# AGENTE (ROBOT DE LIMPIEZA)
# ==========================

class CleaningAgent(CellAgent):
    """
    Robot de limpieza reactivo.

    Reglas:
    - Si la celda actual está sucia: aspira (la limpia).
    - Si la celda actual está limpia: se mueve a una celda vecina aleatoria.
    """

    def __init__(self, model: "CleaningModel", cell):
        """
        Crea un nuevo agente de limpieza.

        Parámetros:
            model: instancia del modelo.
            cell: celda inicial donde empieza el agente.
        """
        super().__init__(model)
        self.cell = cell
        self.moves = 0  # movimientos efectivos (cambio de celda)

    def _isCurrentCellDirty(self) -> bool:
        """Regresa True si la celda actual está sucia."""
        coord = self.cell.coordinate
        return coord in self.model.dirtyCells

    def clean(self) -> None:
        """Limpia la celda actual si está sucia."""
        coord = self.cell.coordinate
        if coord in self.model.dirtyCells:
            self.model.dirtyCells.remove(coord)

    def move(self) -> None:
        """
        Se mueve a una celda vecina aleatoria
        (vecindad de Moore, 8 direcciones).
        """
        oldCell = self.cell
        newCell = self.cell.neighborhood.select_random_cell()
        self.cell = newCell

        # Contamos movimiento solo si cambió de celda
        if self.cell is not oldCell:
            self.moves += 1

    def step(self) -> None:
        """Ejecuta un paso de comportamiento del agente."""
        if self._isCurrentCellDirty():
            self.clean()
        else:
            self.move()


# ==========================
# MODELO
# ==========================

class CleaningModel(mesa.Model):
    """
    Modelo de robots de limpieza sobre una grilla MxN.

    Parámetros:
        nAgents: número de robots.
        width: ancho (M) de la habitación.
        height: alto (N) de la habitación.
        dirtyPercentage: porcentaje inicial de celdas sucias (0.0 a 1.0).
        maxSteps: tiempo máximo (número de pasos de simulación).
    """

    def __init__(
        self,
        nAgents: int = 5,
        width: int = 10,
        height: int = 10,
        dirtyPercentage: float = 0.3,
        maxSteps: int = 200,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)

        self.numAgents = nAgents
        self.width = width
        self.height = height
        self.dirtyPercentage = dirtyPercentage
        self.maxSteps = maxSteps

        # Control de ejecución
        self.running = True
        self.currentStep = 0  # tiempo de simulación (steps)

        # Grid con vecindad de Moore (8 vecinos)
        # capacity=None permite múltiples agentes por celda
        self.grid = OrthogonalMooreGrid(
            (width, height),
            torus=False,
            capacity=None,
            random=self.random,
        )

        # ---------- Inicializar celdas sucias ----------
        allCells = list(self.grid.all_cells.cells)
        totalCells = width * height
        nDirty = int(self.dirtyPercentage * totalCells)
        nDirty = max(0, min(nDirty, totalCells))

        dirtySample = self.random.sample(allCells, k=nDirty)
        # Conjunto de coordenadas sucias
        self.dirtyCells: set[tuple[int, int]] = {
            cell.coordinate for cell in dirtySample
        }

        # ---------- Crear agentes ----------
        # Todos empiezan en la celda [1,1] → coordenada (0,0) en código
        startCell = self.grid[(0, 0)]

        CleaningAgent.create_agents(
            self,
            self.numAgents,
            [startCell] * self.numAgents,
        )

        # ---------- DataCollector ----------
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Step": lambda m: m.currentStep,
                "CleanPercentage": computeCleanPercentage,
                "TotalMoves": computeTotalMoves,
            },
            agent_reporters={
                "Moves": "moves",
            },
        )

        # Registrar estado inicial
        self.datacollector.collect(self)

    # ---------- Utilidades del modelo ----------

    def allClean(self) -> bool:
        """Regresa True si ya no hay celdas sucias."""
        return len(self.dirtyCells) == 0

    # ---------- Paso de simulación ----------

    def step(self) -> None:
        """
        Ejecuta un paso del modelo.

        Se detiene si:
        - Ya no hay celdas sucias, o
        - Se alcanzó el número máximo de pasos (maxSteps).
        """
        if self.allClean():
            self.running = False
            return

        if self.currentStep >= self.maxSteps:
            self.running = False
            return

        # Avanzamos el tiempo
        self.currentStep += 1

        # Todos los agentes actúan en orden aleatorio
        self.agents.shuffle_do("step")

        # Registrar datos del modelo en este step
        self.datacollector.collect(self)

        if self.allClean():
            self.running = False


# ==========================
# VISUALIZACIÓN (GRID + GRÁFICAS)
# ==========================

def agentPortrayal(agent: CleaningAgent) -> AgentPortrayalStyle:
    """
    Define cómo se dibuja cada agente en el grid.

    Rojo si la celda donde está es sucia.
    Verde si está limpia.
    """
    coord = agent.cell.coordinate
    isDirty = coord in agent.model.dirtyCells
    color = "tab:red" if isDirty else "tab:green"
    return AgentPortrayalStyle(color=color, size=50)


# Parámetros del modelo para la UI de Mesa/Solara
modelParams = {
    "nAgents": {
        "type": "SliderInt",
        "value": 5,
        "label": "Número de agentes:",
        "min": 1,
        "max": 200,
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
    "dirtyPercentage": {
        "type": "SliderFloat",
        "value": 0.3,
        "label": "Porcentaje inicial de celdas sucias:",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    "maxSteps": {
        "type": "SliderInt",
        "value": 200,
        "label": "Tiempo máximo (steps):",
        "min": 10,
        "max": 1000,
        "step": 10,
    },
}

# Página 1: gráficas (vs Step)
cleanPlot = make_plot_component("CleanPercentage", page=1)
movesPlot = make_plot_component("TotalMoves", page=1)

# Modelo inicial para la visualización interactiva
cleanModel = CleaningModel(
    nAgents=5,
    width=10,
    height=10,
    dirtyPercentage=0.3,
    maxSteps=200,
)

# Render del espacio (Página 0)
renderer = SpaceRenderer(model=cleanModel, backend="matplotlib").render(
    agent_portrayal=agentPortrayal
)

# SolaraViz:
# - Página 0: grid del modelo
# - Página 1: gráficas CleanPercentage y TotalMoves
page = SolaraViz(
    cleanModel,
    renderer,
    components=[cleanPlot, movesPlot],
    model_params=modelParams,
    name="Reactive Cleaning Robot",
)


# ==========================
# EXPERIMENTOS PARA EL REPORTE
# ==========================

def runExperiment(
    nAgents: int,
    width: int,
    height: int,
    dirtyPercentage: float,
    maxSteps: int,
    seed: int | None = None,
):
    """
    Ejecuta una simulación y regresa las 3 métricas finales.

    Parámetros:
        nAgents: número de agentes.
        width: ancho de la grilla.
        height: alto de la grilla.
        dirtyPercentage: porcentaje inicial de celdas sucias.
        maxSteps: pasos máximos permitidos.
        seed: semilla para el generador aleatorio (opcional).

    Regresa:
        timeUsed: pasos utilizados hasta limpiar todo o llegar al máximo.
        finalCleanPct: porcentaje de celdas limpias al final.
        finalTotalMoves: número total de movimientos.
        model: instancia final del modelo.
    """
    model = CleaningModel(
        nAgents=nAgents,
        width=width,
        height=height,
        dirtyPercentage=dirtyPercentage,
        maxSteps=maxSteps,
        seed=seed,
    )

    # Ejecutar hasta que termine (todo limpio o maxSteps)
    while model.running:
        model.step()

    timeUsed = model.currentStep
    finalCleanPct = computeCleanPercentage(model)
    finalTotalMoves = computeTotalMoves(model)

    return timeUsed, finalCleanPct, finalTotalMoves, model


# ==========================
# MAIN: correr experimentos y gráficas vs número de agentes
# ==========================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Cantidades de agentes a probar
    agentValues = [1, 2, 3, 5, 8, 10, 15, 20, 40, 80, 100]

    rows = []

    for n in agentValues:
        timeUsed, finalCleanPct, finalTotalMoves, _ = runExperiment(
            nAgents=n,
            width=20,
            height=20,
            dirtyPercentage=0.4,
            maxSteps=1000,
            seed=None,
        )
        rows.append({
            "nAgents": n,
            "timeUsed": timeUsed,
            "finalCleanPct": finalCleanPct,
            "finalTotalMoves": finalTotalMoves,
        })

    dfResults = pd.DataFrame(rows)
    print(dfResults)

    # ========= Gráfica 1 — Tiempo vs número de agentes =========
    plt.figure(figsize=(7, 4))
    plt.plot(
        dfResults["nAgents"],
        dfResults["timeUsed"],
        marker="o",
        linestyle="-",
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
        dfResults["nAgents"],
        dfResults["finalTotalMoves"],
        marker="o",
        linestyle="-",
    )
    plt.title("Movimientos totales vs número de agentes")
    plt.xlabel("Número de agentes")
    plt.ylabel("Número total de movimientos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
