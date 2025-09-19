from typing import NamedTuple

import numpy as np


class Obstacle(NamedTuple):
    id: int
    state: np.ndarray  # (6,) array of [x, y, z, vx, vy, vz]
    covariances: np.ndarray  # (6, 6) array of covariances
