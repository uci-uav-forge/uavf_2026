import threading
import time

from shared.pdb_linker import PDBLinker
from perception.lib.util import create_console_logger


PRINT_DELAY = 5  # Frequency to print message in seconds
HORIZ_E = 0.1312335958  # in wh / meter
VERT_E = 0.2624671916  # in wh / meter
MAX_WH = 401.0  # According to the esp calibration (not true scale)
ALT_WH_CALC = False

HORIZ_TIME = 0.237  # in seconds / meter
VERT_TIME = 1.544  # in seconds / meter


class PDBMonitor:
    def __init__(self, commander_node):
        commander_node.log("pdbmonitor starting...")
        self.commander_node = commander_node

        # logging configs
        logs_path = self.commander_node.logs_path

        self._logger = create_console_logger(logs_path, "pdb_logs", False)

        self.linker = PDBLinker(self._logger)

        self.logger_running = False
        self.thread = None

        self.sum_wh = 0.0

        self.rtl_t = -1.0

        self.log("pdb monitor ready...")

    def log(self, *args, **kwargs) -> None:
        """
        Log the given arguments.
        """
        self._logger.info(*args, **kwargs)

    def start_logging(self):
        self.logger_running = True
        self.thread = threading.Thread(target=self._log_runner)
        self.thread.start()

    def stop_logging(self):
        self.log("Stopping logging...")
        self.logger_running = False

    def _log_runner(self):
        last_t = time.time()
        if ALT_WH_CALC:
            self.log(f"[{last_t}]: pdb logger starting to sample values...")
        print_timer = last_t
        while self.logger_running:
            try:
                self.linker.read_voltages()
                now_t = time.time()
                self.log(
                    f"[{now_t}]: {self.linker.voltages} {f', est WH: {self.sum_wh}' if ALT_WH_CALC else ''}"
                )
                if ALT_WH_CALC:
                    bat = self.linker.voltages["MATK"]
                    cur = self.linker.voltages["CUR"]
                    self.sum_wh += (bat * cur) * ((now_t - last_t) / 3600.0)
                    last_t = now_t

                if now_t - print_timer > PRINT_DELAY:
                    bat = self.linker.voltages["MATK"]
                    cur = self.linker.voltages["CUR"]
                    wh = self.linker.voltages["WH"]
                    self.commander_node.log_status(
                        f"BAT {bat:.02f}V | CUR {cur:.02f}A | PWR {(bat * cur):.02f}W | {wh:.01f} Wh{f' | {self.sum_wh:.01f} est WH' if ALT_WH_CALC else ''}",
                        announce=False,
                    )
                    self.commander_node.log_status(
                        f"Time to RTL: {(self.rtl_t / 60.0):.02f} seconds...",
                        announce=False,
                    )
                    print_timer = now_t
            except ValueError as e:
                self.log(e)
            time.sleep(0.5)

    def battery_failsafe(self, horiz_dist, vertic_dist):
        expected_wh = horiz_dist * HORIZ_E + vertic_dist * VERT_E
        return self.linker.voltages["WH"] + expected_wh > MAX_WH

    def rtl_time(self, horiz_dist, vertic_dist):
        self.rtl_t = horiz_dist * HORIZ_TIME + vertic_dist * VERT_TIME
