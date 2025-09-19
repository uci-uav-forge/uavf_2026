# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

# noqa

import cv2

try:
    from Queue import Queue
except ModuleNotFoundError:
    from queue import Queue

import threading


class FrameReader(threading.Thread):
    queues = []
    _running = True
    camera = None

    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera

    def run(self):
        while self._running:
            _, frame = self.camera.read()
            while self.queues:
                queue = self.queues.pop()
                queue.put(frame)

    def addQueue(self, queue):
        self.queues.append(queue)

    def getFrame(self, timeout=None):
        queue = Queue(1)
        self.addQueue(queue)
        return queue.get(timeout=timeout)

    def stop(self):
        self._running = False


class Previewer(threading.Thread):
    window_name = "Arducam"
    _running = True
    camera = None

    def __init__(self, camera, name):
        threading.Thread.__init__(self)
        self.name = name
        self.camera = camera

    def run(self):
        self._running = True
        while self._running:
            cv2.imshow(self.window_name, self.camera.getFrame(2000))
            _keyCode = cv2.waitKey(16) & 0xFF
        cv2.destroyWindow(self.window_name)

    def start_preview(self):
        self.start()

    def stop_preview(self):
        self._running = False


class GstCamera(object):
    frame_reader = None
    cap = None
    previewer = None

    def __init__(self, gst_pipeline: str):
        self.open_camera(gst_pipeline)

    def open_camera(self, pipeline):
        self.cap = cv2.VideoCapture(
            pipeline,
            cv2.CAP_GSTREAMER,
        )
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        if self.frame_reader is None:
            self.frame_reader = FrameReader(self.cap, "")
            self.frame_reader.daemon = True
            self.frame_reader.start()
        self.previewer = Previewer(self.frame_reader, "")

    def getFrame(self):
        return self.frame_reader.getFrame()

    def close(self):
        self.frame_reader.stop()
        self.cap.release()

    def __del__(self):
        self.close()
