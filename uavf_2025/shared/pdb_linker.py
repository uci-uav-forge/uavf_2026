from shared.esp_linker import ESPLinker
import logging
import numpy as np


PDB_VID = "303A"
DEVICE_PHRASE = "PDB_DEV"
SERIAL_BAUD = 115200

BATTERY_CAPACITY_WH = 720


class PDBLinker(ESPLinker):
    def __init__(self, logger: logging.Logger):
        super().__init__(
            logger, PDB_VID, DEVICE_PHRASE, SERIAL_BAUD, silence_failed_connection=True
        )
        self.voltages = {
            "12V": 0.0,
            "5V": 0.0,
            "BAT": 0.0,
            "MATK": 0.0,
            "CUR": 0.00,
            "WH": 0.00,
        }
        self._logger = logger
        self._wh_buf = []

    def regress_data(self) -> int:
        try:
            time_accumulated = [buf[0] for buf in self._wh_buf]
            watt_usage = [buf[1] for buf in self._wh_buf]

            slope, intercept = np.polyfit(watt_usage, time_accumulated, deg=1)
            pred_time = slope * BATTERY_CAPACITY_WH + intercept
        except TypeError:
            pred_time = -1

        return pred_time

    def read_voltages(self):
        resp = self.sendmessage("V")

        self._logger.debug(f"Read voltage output: {resp}")
        # remove the OK
        resp = resp[3:]

        # split it to extract info
        resp = resp.split(",")
        for voltage in resp:
            if "12V" in voltage:
                self.voltages["12V"] = float(voltage.replace("12V ", ""))
            elif "5V" in voltage:
                self.voltages["5V"] = float(voltage.replace("5V ", ""))
            elif "BAT" in voltage:
                self.voltages["BAT"] = float(voltage.replace("BAT ", ""))
            elif "MATK_CUR" in voltage:
                self.voltages["CUR"] = float(voltage.replace("MATK_CUR ", ""))
            elif "MATK_VOL" in voltage:
                self.voltages["MATK"] = float(voltage.replace("MATK_VOL ", ""))
            elif "WH" in voltage:
                self.voltages["WH"] = float(voltage.replace("WH ", ""))
