from std_srvs.srv import Empty
import os
from enum import Enum
from typing import TYPE_CHECKING
from datetime import datetime
from pathlib import Path

if TYPE_CHECKING:  # Prevents circular import
    pass
from shared.payload import Payload
from perception.lib.util import create_console_logger


class PayloadManager:
    class PayloadType(Enum):
        NO_PAYLOAD = 0
        GAZEBO_PAYLOAD = 1
        PAYLOAD_MODULE = 2

    def __init__(self, commander_node):
        self.commander_node = commander_node
        src_path = Path(__file__).absolute().parent.parent.parent.parent
        self.filepath = src_path / "uavf_2025/uavf_2025/shared/logs"
        self._logger = create_console_logger(
            self.filepath,
            "payload_logs_{:%Y-%m-%d-%m-%s}".format(datetime.now()),
            False,
        )

        ### determine what payload type to use
        if os.environ.get("PAYLOAD_ACTIVE", False):
            self.payload = Payload(self._logger)
            self.payload_type = PayloadManager.PayloadType.PAYLOAD_MODULE
            self.commander_node.log("Real payload module ready.")
        else:
            # Try to make gazebo payload
            self.drop_client = commander_node.create_client(Empty, "/uavf_drop_srv")

            if not self.drop_client.wait_for_service(timeout_sec=5):
                # if gazebo payload service isn't running
                commander_node.log(
                    "Couldn't connect to drop service. Check if the service is actually running."
                )
                self.payload_type = PayloadManager.PayloadType.NO_PAYLOAD
            else:
                self.commander_node.log("Gazebo payload ready.")
                self.payload_type = PayloadManager.PayloadType.GAZEBO_PAYLOAD

    def drop_payload(self):
        """
        Drops the payload. Function will hang until payload is dropped and drone can safely keep moving.
        """
        if self.payload_type == PayloadManager.PayloadType.GAZEBO_PAYLOAD:
            self.commander_node.log("Dropping gazebo payload...")
            return self.drop_client.call_async(Empty.Request()).result()
        elif self.payload_type == PayloadManager.PayloadType.PAYLOAD_MODULE:
            self.commander_node.log("Droping real payload...")
            self.payload.drop()
            self.commander_node.log("Attempting retract...")
            self.payload.retract()  # Will only work once all payloads are dropped
            return True
        else:
            self.commander_node.log("Drop called but no payload set up.")
            return None

    def reset(self):
        if self.payload_type == PayloadManager.PayloadType.PAYLOAD_MODULE:
            self.payload.clear_ctr()
