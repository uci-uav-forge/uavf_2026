from ultralytics import YOLO
from ultralytics.engine.results import Results
import os
from pathlib import Path
from perception.types import Bbox2D, Image, TargetBBox
import numpy as np
import subprocess
from line_profiler import profile

CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

MAX_NUM_CLASSES = 15


def check_jetson_clocks():
    try:
        subprocess.run(
            ["sudo", "-S", "jetson_clocks"],
            input="copp\n",  # password for sudo
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


class TargetDetector:
    CLASSES = []
    _img_id = 0

    def __init__(self):
        # if jetson_clocks returns a good return code, use diff model
        if check_jetson_clocks():
            # obtained with yolo export model=trained_yolo11n.pt format=tensorrt imgsz=1280
            # when we make new engine files, IT NEEDS TO BE DONE ON THE JETSON ITSELF
            self.model = YOLO(CURRENT_DIR / "trained_yolo11n.engine", task="detect")
        else:
            self.model = YOLO(CURRENT_DIR / "trained_yolo11n.torchscript")
        # warm up model on random image. If we don't do this, the first prediction is really slow
        self.model.predict(
            np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8), imgsz=640
        )

    @profile
    def detect(self, image: Image) -> list[TargetBBox]:
        results: Results = list(
            self.model.track(image.get_array(), verbose=False, imgsz=640)
        )[0]

        detection = []
        for i in range(len(results)):  # type: ignore
            result: Results = results[i]
            cls_prob = np.zeros(15)
            if result.boxes is None:  # this never happens, just to make types work out
                continue

            cls_prob[int(result.boxes.cls.cpu().item())] = (
                result.boxes.conf.cpu().item()
            )
            if result.boxes.id is None:
                continue

            detection.append(
                TargetBBox(
                    Bbox2D(*result.boxes.xywh[0].cpu().numpy()),
                    cls_prob,
                    TargetDetector._img_id,
                    int(result.boxes.id.cpu().numpy()),
                )
            )

        TargetDetector._img_id += 1
        return detection
