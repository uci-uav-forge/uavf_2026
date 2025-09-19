"""
payload.py
v 0.1 by Lucas Li
"""

import time
import os
import logging
from pathlib import Path
from perception.lib.util import create_console_logger
from datetime import datetime
from shared.esp_linker import ESPLinker


PAYLOAD_VID = "303A"
DEVICE_PHRASE = "PL"
MAC_ADDRESS = ""
SERIAL_BAUD = 115200


class Payload(ESPLinker):
    """
    Payload Module Class. Used to control the ESP32 C3
    """

    def __init__(self, logger: logging.Logger, drop_time: int = 25):
        super().__init__(logger, PAYLOAD_VID, DEVICE_PHRASE, SERIAL_BAUD)

        # Parameters
        self._drop_time = drop_time
        self._payload_count = 2
        self._logger = logger

    @property
    def remaining(self):
        """Returns how many Payloads are left."""
        return 2 - self._payload_count

    def __repr__(self):
        """If you ever feel useless, just know that this function exists."""
        return f"Serial Device {self._device_phrase} at {self._port_addr}, \
                {self._payload_count} Payloads, Drop time {self._drop_time}"

    def drop(self) -> bool:
        """Drop Function, Returns bool for completion, prevents over-extension."""
        if self._payload_count > 0:
            response = self.sendmessage("D\n")
            if "Drop" in response.strip():
                self._payload_count -= 1
                time.sleep(self._drop_time)
                self._logger.info("Dropping payload...")
                return True
            return False
        return False

    def load(self) -> bool:
        """Load Function"""
        if self._payload_count > 0:
            response = self.sendmessage("L\n")
            if "Load" in response.strip():
                self._logger.info("Loading payload...")
                return True
            return False
        return False

    def retract(self) -> bool:
        """Retract Function, Returns bool for completion, prevents over-retraction, can only retract after all payloads dropped."""
        self._logger.info(f"payload count = {self._payload_count}")
        if self._payload_count < 1:
            self._logger.info("retracting...")
            # Only allow retraction after both payloads have been dropped
            response = self.sendmessage("R\n")
            print(response)
            if "Retracted" in response.strip():
                self._payload_count += 2
                self._logger.info("Retracting payload...")
                self.clear_ctr()
                return True
            return False
        return False

    def clear_ctr(self) -> bool:
        # Only allow retraction after both payloads have been dropped
        response = self.sendmessage("C\n")
        print(response)
        if "Counter Cleared" in response.strip():
            self._payload_count = 2
            self._logger.info("Clearing esp counter...")
            return True
        return False

    def ping(self) -> bool:
        """Mac Address is being used as a ping"""
        response = self.sendmessage("M\n")
        self._logger.info(response)
        return bool(response)


if __name__ == "__main__":
    port = os.environ.get("PAYLOAD_ACTIVE", False)
    if port:
        src_path = Path(__file__).absolute().parent.parent.parent.parent
        filepath = src_path / "uavf_2025/uavf_2025/shared/logs"
        _logger = create_console_logger(
            filepath,
            "payload_logs_{:%Y-%m-%d-%m-%s}".format(datetime.now()),
            False,
        )
        test_payload = Payload(_logger, 5)
        _logger.info(f"Pinging payload: {test_payload.ping()}")
        test_payload.drop()
        test_payload.drop()
        test_payload.retract()
        del test_payload
    else:
        print("Enviorment variable not found or turned off!")
