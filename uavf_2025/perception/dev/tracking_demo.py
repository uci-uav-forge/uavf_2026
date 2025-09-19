from perception.camera import make_gimballed_camera
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class PIDController:
    def __init__(self, P_gain, I_gain, D_gain):
        self.P = P_gain
        self.I = I_gain
        self.D = D_gain
        self.total_err = 0
        self.prev_err = 0

    def step(self, err):
        """
        Returns control input
        """
        self.total_err += err
        err_diff = err - self.prev_err
        self.prev_err = err
        return self.P * err + self.I * self.total_err - self.D * err_diff


CURRENT_DIR = Path(__file__).absolute().parent

if __name__ == "__main__":
    camera = make_gimballed_camera(log_dir=None)
    model = YOLO(f"{CURRENT_DIR}/../lib/coco_yolo11n.pt")
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1900, 1000)
    tracked_id = None
    camera.point_center()
    pid_yaw = PIDController(5e-2, 0, 1e-2)
    pid_pitch = PIDController(5e-2, 0, 1e-2)
    frame_cnt = 0
    while True:
        frame = camera.take_image()
        if frame is None:
            frame_arr = np.zeros((1080, 1920, 3))
            cv2.putText(
                frame_arr,
                "No image",
                (10, 320),
                cv2.FONT_HERSHEY_COMPLEX,
                2,
                (255, 255, 255),
            )
        else:
            frame_cnt += 1
            if frame_cnt % 30 == 0:
                camera.do_autofocus()
            frame_arr = np.ascontiguousarray(frame.get_array(), dtype=np.uint8)
            track_res = model.track(frame_arr.copy(), verbose=False)[0]
            if track_res.boxes is not None and track_res.boxes.id is not None:
                for box, id, cls in zip(
                    track_res.boxes.xyxy, track_res.boxes.id, track_res.boxes.cls
                ):
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0)
                    if model.names[int(cls)] == "person":
                        if tracked_id is None:
                            tracked_id = id.item()
                    if id.item() == tracked_id:
                        if model.names[int(cls)] != "person":
                            tracked_id = None
                            camera.point_center()
                            continue
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        err_x = cx - frame.shape[1] // 2
                        err_y = cy - frame.shape[0] // 2
                        yaw, pitch, roll = camera.get_attitude()
                        yaw_input = pid_yaw.step(err_x)
                        pitch_input = pid_pitch.step(err_y)
                        camera.set_absolute_position(
                            yaw - yaw_input, pitch - pitch_input
                        )
                        color = (0, 0, 255)
                        cv2.putText(
                            frame_arr,
                            f"yaw: {yaw:.02f}, pitch: {pitch:.02f}, yaw_input: {yaw_input:.02f}, pitch_input: {pitch_input:.02f}",
                            (0, 25),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 0, 0),
                        )
                        cv2.putText(
                            frame_arr,
                            f"err_x: {err_x:.02f}, err_y: {err_y:.02f}",
                            (0, 50),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 0, 0),
                        )
                    cv2.rectangle(frame_arr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_arr,
                        f"{model.names[int(cls)]} ({id.item()})",
                        (x1, y1),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        color,
                    )

        cv2.imshow("image", frame_arr)
        key = cv2.waitKey(1)
        if 0 <= key - ord("0") <= 9:
            tracked_id = key - ord("0")
