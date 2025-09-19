import serial.tools.list_ports
import subprocess
import logging


def find_pixhawk_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "pixhawk" in port.description.lower():
            return port.device  # e.g., '/dev/ttyUSB0' or 'COM3'
    return None


def run_bash_command(command: str, log_file: str = "command_output.log"):
    # Set up logging
    logging.basicConfig(
        filename=log_file,
        filemode="a",  # append mode
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Running command: {command}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            logging.info("STDOUT:\n" + result.stdout.strip())
        if result.stderr:
            logging.error("STDERR:\n" + result.stderr.strip())

    except Exception as e:
        logging.exception(f"Exception while running command: {e}")


def start_mavros():
    port = find_pixhawk_port()
    if port is not None:
        print(port)
        run_bash_command(
            f"ros2 launch mavros apm.launch fcu_url:={port}:921600/?sysid=1"
        )
    else:
        run_bash_command(
            "ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://127.0.0.1:14445@14445/?sysid=1"
        )


if __name__ == "__main__":
    start_mavros()
