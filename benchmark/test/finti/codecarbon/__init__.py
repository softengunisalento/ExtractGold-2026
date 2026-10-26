"""CodeCarbon FINTO, solo per la prova simulata: energia proporzionale alla durata (potenze fisse; a riposo,
cioè nella finestra chiamata "riposo", la metà)."""
import time

__version__ = "finto"

POTENZE_W = {"gpu": 300.0, "cpu": 100.0, "ram": 10.0}


class EmissionsData:
    def __init__(self, durata: float, fattore: float = 1.0):
        self.duration = durata
        self.gpu_energy = fattore * POTENZE_W["gpu"] * durata / 3600 / 1000
        self.cpu_energy = fattore * POTENZE_W["cpu"] * durata / 3600 / 1000
        self.ram_energy = fattore * POTENZE_W["ram"] * durata / 3600 / 1000
        self.energy_consumed = self.gpu_energy + self.cpu_energy + self.ram_energy
        self.emissions = self.energy_consumed * 0.3


class CPU:
    _mode = "intel_rapl"


class GPU:
    _mode = "nvml"


class RAM:
    _mode = "stima"


class EmissionsTracker:
    def __init__(self, tracking_mode="machine", save_to_file=True, log_level="info", allow_multiple_runs=False,
                 project_name="codecarbon"):
        self.tracking_mode = tracking_mode
        self._hardware = [CPU(), GPU(), RAM()]
        self.final_emissions_data = None
        self._t = self._task = None

    def start(self):
        self._t = time.perf_counter()

    def stop(self):
        self.final_emissions_data = EmissionsData(time.perf_counter() - self._t)
        return self.final_emissions_data.emissions

    def start_task(self, task_name=""):
        self._task = time.perf_counter()
        self._nome = task_name

    def stop_task(self):
        return EmissionsData(time.perf_counter() - self._task, 0.5 if self._nome == "riposo" else 1.0)
