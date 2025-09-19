import numpy as np
import torch
from torchvision.transforms import functional as F

from perception.camera.project import project
from perception.types import Bbox2D, CameraPose

from .types import Obstacle


class ParticleFilter:
    """
    NOTE: Make an ABC that this inherits if we use more than one localization method.
    """

    DEFAULT_REGION_CENTER = torch.tensor([0, 0, 0], dtype=torch.float16)
    DEFAULT_REGION_SIZE = torch.tensor([1, 1, 1], dtype=torch.float16)

    OUT_OF_FRAME_MEAN = 0.3
    OUT_OF_FRAME_STD_DEV = 0.2

    def __init__(
        self,
        region_center: torch.Tensor = DEFAULT_REGION_CENTER,
        region_size: torch.Tensor = DEFAULT_REGION_SIZE,
        num_particles: int = 10000,
        score_threshold: float = 0.5,
        location_noise_std_dev=0.1,
        velocity_noise_std_dev=0.1,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        self._num_particles = num_particles
        self._score_threshold = score_threshold

        self._num_dims = len(region_center)

        self._region_center = region_center.to(device)
        self._region_size = region_size.to(device)
        if region_center.shape != region_size.shape:
            raise ValueError("region_center and region_size must have the same shape")

        self._location_noise_std_dev = location_noise_std_dev
        self._velocity_noise_std_dev = velocity_noise_std_dev
        if not location_noise_std_dev > 0 or not velocity_noise_std_dev > 0:
            raise ValueError("Noise standard deviations must be positive")

        self._device = device

        # (N, 6) tensor, of location and velocity
        self._particles = self._make_particles()

    def update(
        self,
        detections: list[list[Bbox2D]],
        poses: list[CameraPose],
        frame_sizes: list[tuple[int, int]],
        step_time: float,
    ) -> None:
        scores: list[torch.Tensor] = []

        for batch, pose, size in zip(detections, poses, frame_sizes):
            mask = self._make_mask(batch, size)
            projection, is_valid = self._get_particles_projection(pose, size)

            # (N,) tensor
            score = self._get_particles_likelihood(mask, projection, is_valid)
            scores.append(score)

        # Take a particle's best score from all cameras
        # to account for objects being in one camera but not the other
        composite_scores = torch.max(torch.stack(scores), dim=0).values

        self._resample(composite_scores)
        self._step_particles(step_time)

    def reset(self) -> None:
        self._particles = self._make_particles()

    def get_estimates(self) -> list[Obstacle]:
        return [
            Obstacle(0, particle, np.zeros((6, 6)))
            for particle in self._particles.cpu().numpy()
        ]

    def get_particles(self) -> torch.Tensor:
        return self._particles.clone()

    def _make_particles(self) -> torch.Tensor:
        particles = torch.rand(
            (self._num_particles, self._num_dims * 2),
            dtype=torch.float16,
            device=self._device,
        )

        # Scale and shift the particle locations to the region
        particles[:, :3] *= self._region_size
        particles[:, :3] += self._region_center - self._region_size / 2

        return particles

    def _make_mask(
        self, detections: list[Bbox2D], img_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Makes a mask by filling in the bounding boxes of the detections,
        then blurring the mask.
        """
        # Torchvision expects CHW, but the mask is greyscale
        mask = torch.zeros(img_size, dtype=torch.float32, device=self._device)
        if len(detections) == 0:
            return mask

        bbox_corners = torch.tensor(
            [[d.x, d.y, d.x, d.y] for d in detections],
            dtype=torch.float32,
            device=self._device,
        )

        bbox_sizes = torch.tensor(
            [[d.width, d.height] for d in detections],
            dtype=torch.float32,
            device=self._device,
        )

        bbox_corners[:, :2] -= bbox_sizes
        bbox_corners[:, 2:] += bbox_sizes

        bbox_corners[:, :2] *= torch.tensor(
            img_size[::-1], dtype=torch.float32, device=self._device
        )

        bbox_corners[:, 2:] *= torch.tensor(
            img_size[::-1], dtype=torch.float32, device=self._device
        )

        for x_min, y_min, x_max, y_max in bbox_corners:
            mask[
                int(y_min) : int(y_max),
                int(x_min) : int(x_max),
            ] = 1

        return F.gaussian_blur(mask.unsqueeze(0), kernel_size=[5, 5]).squeeze()

    def _get_particles_projection(
        self, pose: CameraPose, frame_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the image coordinates of particles and a mask of whether they
        are in the image (i.e., within frame and in front of the frustrum).

        Returns
        -------
        pts2 : 2D torch.Tensor
            Image coordinates of N points stored in a tensor of shape (N, 2).

        in_frame_arr : torch.Tensor
            Tensor of shape (N,) containing 1 if the point is inside the frame.
        """
        coords_2d, in_frame = project(self._particles.T[:3], pose, frame_size)

        return coords_2d.round().T.to(torch.int), in_frame

    def _get_particles_likelihood(
        self, mask: torch.Tensor, coordinates: torch.Tensor, is_valid: torch.Tensor
    ) -> torch.Tensor:
        """
        Gets the likelihood of particles based on where they are in the mask.

        NOTE: This method will modify the coordinates tensor in-place.
        """
        # Need to clamp so the indexing doesn't error
        torch.clamp(
            coordinates,
            torch.tensor([0, 0], device=self._device),
            torch.tensor(mask.shape[::-1], device=self._device) - 1,
            out=coordinates,
        )

        scores = mask[coordinates[:, 1], coordinates[:, 0]]

        # If the particle is out of frame, give it a random low score
        scores[~is_valid] = (
            torch.randn((len(scores[~is_valid]),)).to(self._device)
            * self.OUT_OF_FRAME_STD_DEV
            + self.OUT_OF_FRAME_MEAN
        )

        return scores

    def _resample(self, scores: torch.Tensor) -> None:
        """
        Resamples the particles based on the scores,
        replacing the filtered ones in-place.
        """
        assert scores.shape == (self._num_particles,)

        is_good = scores > self._score_threshold
        is_good_indices = torch.nonzero(is_good).squeeze()
        good_count = int(is_good.sum().item())

        is_bad = ~is_good
        is_bad_indices = torch.nonzero(is_bad).squeeze()
        bad_count = int(is_bad.sum().item())

        # Select random good particles to replace bad ones
        replacement_indicies = is_good_indices[
            torch.randint(0, good_count, (bad_count,))
        ]

        # Pick out the chosen good particles and add some noise to them
        replacement_particles = self._particles[replacement_indicies]

        assert replacement_particles.shape == (bad_count, self._num_dims * 2)

        self._particles[is_bad_indices] = replacement_particles
        self._particles += self._make_noise()

    def _make_noise(self) -> torch.Tensor:
        """
        Generates Gaussian noise for new particles based on
        parameterized standard deviations.
        """
        noise = torch.randn(
            (self._num_particles, self._num_dims * 2),
            dtype=torch.float16,
            device=self._device,
        )

        noise[:, : self._num_dims] *= self._location_noise_std_dev
        noise[:, self._num_dims :] *= self._velocity_noise_std_dev

        return noise

    def _step_particles(self, step_time: float) -> None:
        """
        Steps the particles forward in time.
        """
        distance = self._particles[:, self._num_dims :] * step_time
        self._particles[:, : self._num_dims] += distance
