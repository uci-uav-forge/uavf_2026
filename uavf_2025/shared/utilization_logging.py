from concurrent.futures import ThreadPoolExecutor
import csv
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path
from time import sleep, time

import numpy as np
from psutil import cpu_percent, virtual_memory

import pandas as pd
from matplotlib import pyplot as plt


class UtilizationLogger:
    """
    This class is responsible for logging CPU and memory utilization to a csv file.

    TODO: Log GPU utilization as well.
    """

    def __init__(self, csv_path: Path, period: float = 0.2):
        self.period = period

        self._prepare_csv(csv_path)

        # This ensures that there's no race condition
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.logging = False

    def _prepare_csv(self, path: Path):
        """
        Prepares the CSV file by creating it and writing the header dependigng on the number of CPUs.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(path, "w")
        self.writer = csv.writer(self.file)

        num_cpus = cpu_count()
        row = chain(["time", "memory", "cpu"], (f"cpu{i}" for i in range(num_cpus)))

        self.writer.writerow(row)
        self.file.flush()

    def log(self):
        """
        Logs the current CPU and memory utilization.
        """
        timestamp = time()
        memory = virtual_memory().percent
        cpu_avg = cpu_percent()
        cpus = cpu_percent(percpu=True)

        row = chain([timestamp, memory, cpu_avg], cpus)
        self.writer.writerow(row)
        self.file.flush()

        return row

    def start(self):
        """
        Starts the logging process.
        """
        self.logging = True
        self.pool.submit(self._log_periodically)

    def stop(self):
        """
        Stops the logging process.
        """
        self.logging = False

    def _log_periodically(self):
        """
        Logs the utilization periodically by submitting the log method to the thread pool,
        then sleeping for the period.

        This blocks, so it should be run in a separate thread.
        This is done in `start`, which submits it to the thread pool.
        """
        while self.logging:
            # Execution is asynchronous, so sleeping directly for the period
            # is acceptable here.
            self.pool.submit(self.log)
            sleep(self.period)

    @staticmethod
    def visualize_log_file(fp: Path, offset_time: bool = True):
        """
        Visualizes the CPU and memory utilization from a CSV log file.
        Parameters
        ----------
        fp : Path
            The path to the CSV file containing the utilization log.
        offset_time : bool
            If True, offsets the time to start from 0.
        """
        assert fp.is_file() and fp.suffix == ".csv", "Must be a CSV file."

        df = pd.read_csv(fp)

        num_cpus = len(df.columns) - 3

        time_range = np.array(df["time"].values) - (
            df["time"].values[0] if offset_time else 0
        )

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time_range, df["memory"], label="Memory Utilization (%)")
        plt.ylim((0, 100))
        plt.title("Memory Utilization")
        plt.xlabel("Time (s)")
        plt.ylabel("Memory Utilization (%)")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        for i in range(num_cpus):
            plt.plot(time_range, df[f"cpu{i}"], label=f"CPU {i} Utilization (%)")
        plt.title("CPU Utilization")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Utilization (%)")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    log_path = Path("utilization_log.csv")

    logger = UtilizationLogger(log_path, period=0.5)
    logger.start()

    sleep(5)

    logger.stop()

    UtilizationLogger.visualize_log_file(log_path)
    log_path.unlink(missing_ok=True)
