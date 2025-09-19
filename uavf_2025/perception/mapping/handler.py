import logging
from pathlib import Path
from perception.types import Image, CameraPose
from .dumb_mapper import DumbMapperSystem
from perception.lib.util import create_console_logger, ProcessMemLogger
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from dataclasses import dataclass
from time import sleep
import numpy as np


class MappingStatus_ToInternal:
    pass


class MappingStatus_FromInternal:
    pass


@dataclass
class MT_Quit(MappingStatus_ToInternal):
    pass


@dataclass
class MT_Start(MappingStatus_ToInternal):
    pass


@dataclass
class MT_Stop(MappingStatus_ToInternal):
    pass


@dataclass
class MT_Stitch(MappingStatus_ToInternal):
    image: Image
    camera_pose: CameraPose


@dataclass
class MF_AcceptingStitch(MappingStatus_FromInternal):
    pass


@dataclass
class MT_RequestCoverageImage(MappingStatus_ToInternal):
    pass


@dataclass
class MF_CoverageImage(MappingStatus_FromInternal):
    image: np.ndarray


class MapperHandler:
    """Multi-processing handler for mapper
    all methods except _loop are external (main perception thread)
    everything in _loop are internal (mapping thread)
    """

    LOG_MSG_PREFIX_EXT = "[Perception/MapperHandler]: "
    LOG_MSG_PREFIX_INT = "[Mapping/MapperHandler]: "

    def __init__(
        self,
        console_logger: logging.Logger,
        logging_path: Path,
        enable_benchmarking: bool,
        mapping_roi: np.ndarray,
    ):
        self._external_conn, internal_conn = Pipe()
        self._console_logger = console_logger
        self._process = Process(
            target=self._loop,
            args=(internal_conn, logging_path, enable_benchmarking, mapping_roi),
        )
        self._process.start()

        self._coverage_image: np.ndarray = np.zeros((1000, 1000))

    def __del__(self):
        self._console_logger.info(self.LOG_MSG_PREFIX_EXT + " Terminating process")
        self._external_conn.send(MT_Quit())
        self._process.join()
        self._external_conn.close()

        self._console_logger.info(self.LOG_MSG_PREFIX_EXT + " Terminated process")

    def start(self) -> None:
        self._console_logger.info(self.LOG_MSG_PREFIX_EXT + " Start")
        self._external_conn.send(MT_Start())

    def stop(self) -> None:
        self._console_logger.info(self.LOG_MSG_PREFIX_EXT + " Stop")
        self._external_conn.send(MT_Stop())

    def get_coverage_image(self) -> np.ndarray:
        self._console_logger.debug(
            self.LOG_MSG_PREFIX_EXT + " Requesting coverage image"
        )
        self._external_conn.send(MT_RequestCoverageImage())
        if self._external_conn.poll():
            mf: MappingStatus_FromInternal = self._external_conn.recv()
            if isinstance(mf, MF_CoverageImage):
                self._coverage_image = mf.image
                self._console_logger.debug(
                    self.LOG_MSG_PREFIX_EXT + " Received coverage image"
                )
            else:
                raise RuntimeError("Unexpected message type received")

        return self._coverage_image

    def update(self, image: Image, camera_pose: CameraPose):
        self._external_conn.send(MT_Stitch(image=image, camera_pose=camera_pose))

    @staticmethod
    def _loop(
        internal_conn: Connection,
        logging_path: Path,
        enable_benchmarking: bool,
        mapping_roi: np.ndarray,
    ):
        """
        internal
        """
        if not logging_path.exists():
            logging_path.mkdir(parents=True)
        console_logger = create_console_logger(logging_path, "mapping_thread")
        mem_logger = ProcessMemLogger()

        console_logger.info(MapperHandler.LOG_MSG_PREFIX_INT + " Process started")

        mapper = DumbMapperSystem(console_logger, logging_path, mapping_roi)

        mt: MappingStatus_ToInternal
        while mt := internal_conn.recv():
            match mt:
                case MT_Quit():
                    console_logger.info(
                        MapperHandler.LOG_MSG_PREFIX_INT
                        + " Terminating process received"
                    )
                    internal_conn.close()
                    break
                case MT_Start():
                    console_logger.info(MapperHandler.LOG_MSG_PREFIX_INT + " Waiting")
                    sleep(15)
                case MT_Stop():
                    mapper.save_map()
                    mapper.clear()
                case MT_RequestCoverageImage():
                    coverage_map = mapper.get_coverage_image()
                    internal_conn.send(MF_CoverageImage(coverage_map))
                case MT_Stitch(image, camera_pose):
                    mapper.stitch_with_optimization(image, camera_pose)
            if enable_benchmarking:
                console_logger.debug("TICK")
                console_logger.debug(mem_logger())

        console_logger.info(MapperHandler.LOG_MSG_PREFIX_INT + " Process ended")
