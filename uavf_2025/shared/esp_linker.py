from serial import Serial
import serial
from serial.tools import list_ports
import logging
import traceback
from datetime import datetime

connected_addresses = []


class ESPLinker:
    def __init__(
        self,
        logger: logging.Logger,
        vid: str,
        device_phrase: str,
        serial_baud: int = 115200,
        silence_failed_connection=False,
    ):
        self.voltages = {"12V": 0.0, "5V": 0.0, "BAT": 0.0, "MATK": 0.0}
        self._vid = vid
        self._device_phrase = device_phrase
        self._baud = serial_baud
        self._logger = logger
        self._silence_failed_connection = silence_failed_connection
        try:
            self._port = self._scan_for_ports()
            self._port_addr = self._port.portstr
            connected_addresses.append(self._port_addr)
            logger.info(f"blacklisted: {self._port_addr}")
        except serial.SerialException as s:
            # silence failed exception if specified
            if not silence_failed_connection:
                raise s

    def _scan_for_ports(self) -> serial.Serial:
        port_addr = None
        serial_port = None
        ports = list_ports.comports()
        for device in ports:
            if (
                device.vid
                and "{:04X}".format(device.vid) == self._vid
                and ("/dev/" + device.name) not in connected_addresses
            ):
                port_addr = "/dev/" + device.name
                try:
                    # don't need to call open because this auto opens if port is given.
                    serial_port = Serial(
                        port_addr, self._baud, write_timeout=5, timeout=1
                    )
                    self._logger.debug(f"Attempting to connect to com port {port_addr}")
                    if not self._verify_connection(serial_port):
                        self._logger.debug(
                            "Connected to wrong com port, disconnecting..."
                        )
                        serial_port.close()
                    else:
                        break

                except serial.SerialException:
                    self._logger.critical(
                        f"Could not open Serial interface to {self._device_phrase} at {port_addr}"
                    )
                    self._logger.critical(traceback.format_exc())
                else:
                    self._logger.info(
                        f"Opened serial port on {port_addr} to device: {self._device_phrase}"
                    )
        if port_addr is None or serial_port is None:
            self._logger.critical(
                f"No valid port found with device phrase {self._device_phrase}"
            )
            raise serial.SerialException(
                f"No valid port found with device phrase {self._device_phrase}"
            )

        return serial_port

    def _read_response(self, port: serial.Serial):
        resp = port.readline().decode()
        while "OK" not in resp:
            resp = port.readline().decode()
        return resp

    def _verify_connection(self, port: serial.Serial):
        # method used to verify that the PDB is actually the PDB
        port.write("O".encode())
        resp = self._read_response(port)

        self._logger.debug(f"Verified connection to {self._device_phrase}: {resp}")
        if "_DEV" in resp:
            self._logger.warning(
                f"The deivce {self._device_phrase} seems to be running a dev version of the firmware. Functionality can not be guaranteed."
            )
        if self._device_phrase in resp:
            self._logger.info(
                f"Found Device with correct binding phrase {self._device_phrase}"
            )
            return True
        return False

    def sendmessage(self, message) -> str:
        """Sends messages to Payload ESP32. Used by other functions."""
        try:
            if self._port and self._port.is_open:
                self._port.write(message.encode())
                response_lines = self._port.readlines()
                self._logger.debug(f"{datetime.now()}: {response_lines}")
                response = ""
                for line in response_lines:
                    response += str(line.decode())
                self._port.flush()
                return response
            else:
                raise serial.SerialException("Trying to send message over closed port.")
        except serial.SerialException as e:
            return str(e)

    def __del__(self):
        # free com port
        self._port.close()
        self._logger.info(
            f"Port closed, device: {self._device_phrase} at {self._port_addr}"
        )
