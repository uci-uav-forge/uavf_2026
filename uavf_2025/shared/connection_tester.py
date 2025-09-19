from shared import PayloadManager
from shared import PDBMonitor
from time import sleep


class DummyCommanderNode:
    def log(self, msg):
        print(msg)

    def log_status(self, msg, announce=True):
        print(msg)


if __name__ == "__main__":
    dcn = DummyCommanderNode()
    payload = PayloadManager(dcn)
    pdbm = PDBMonitor(dcn)

    sleep(2)

    pdbm.start_logging()

    payload.reset()

    payload.drop_payload()
    sleep(1)
    payload.drop_payload()

    sleep(120)

    pdbm.stop_logging()
