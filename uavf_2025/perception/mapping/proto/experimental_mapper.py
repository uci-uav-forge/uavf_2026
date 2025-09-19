"""
Experimenting ground for mapping techniques to be later integrated at mapping/internal.py
"""

import numpy as np
import cv2
from typing import Optional, Any


class ExperimentalMapper_KeyPointWarping:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher()

        self.img_agg: Optional[cv2.typing.MatLike] = None
        self.prev_kpD: Optional[list[cv2.KeyPoint]] = None
        self.prev_desD: Optional[np.ndarray] = None

    def clear(self) -> None:
        """
        Clears existing stored/aggregated map
        """
        self.img_agg = None
        self.prev_kpD: Optional[list[cv2.KeyPoint]] = None
        self.prev_desD: Optional[np.ndarray] = None

    def get_raw_map(self) -> Optional[cv2.typing.MatLike]:
        return self.img_agg

    def stitch(self, img_delta) -> Any:
        """
        self.img_agg: aggregated image (avoid warping this)
        img_delta: new single image frame
        """
        if self.img_agg is None:
            self.prev_kpD, self.prev_desD = self.sift.detectAndCompute(img_delta, None)
            self.img_agg = img_delta
            return True

        self.img_delta = img_delta

        print(
            "MAPPING [Internal]: Detecting"
        )  # detect new keypoints and compute descriptors
        kpD, desD = self.sift.detectAndCompute(
            cv2.cvtColor(img_delta, cv2.COLOR_RGBA2GRAY), None
        )

        print("MAPPING [Internal]: Matching")  # match common descriptors
        if desD is None:
            print("MAPPING [Internal]: Failed to find any keypoints and descriptors")
            return False

        print(self.prev_desD.shape, desD.shape)
        matches = self.matcher.match(self.prev_desD, desD)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            print("MAPPING [Internal]: Less than 4 keypoints/descriptors matched")
            return False

        # find associated local keypoints from matched descriptors
        ptsA = np.array(
            [self.prev_kpD[m.queryIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)
        ptsD = np.array(
            [kpD[m.trainIdx].pt for m in matches], dtype=np.float32
        ).reshape(-1, 1, 2)

        print("MAPPING [Internal]: Finding homography")  # find homography
        H, _point_mask = cv2.findHomography(ptsD, ptsA, cv2.RANSAC, 5.0)

        if H is None:
            print(
                f"MAPPING [Internal]: Failed to find homography with {len(matches)} matches"
            )
            return False

        hA, wA = self.img_agg.shape[:2]
        hD, wD = img_delta.shape[:2]

        print(
            "MAPPING [Internal]: Translating homography"
        )  # find translated homography (if corners have negative coordinates)
        cornersD = np.array(
            [[0, 0], [0, hD], [wD, hD], [wD, 0]], dtype=np.float32
        ).reshape(-1, 1, 2)
        cornersD_raw_t = cv2.perspectiveTransform(cornersD, H)

        img_t = (
            -min(cornersD_raw_t[:, :, 0].min() - 1, 0),
            -min(cornersD_raw_t[:, :, 1].min() - 1, 0),
        )
        img_ti = (int(np.ceil(img_t[0])), int(np.ceil(img_t[1])))
        mat_t = np.array(
            [
                [1.0, 0.0, -img_t[0]],
                [0.0, 1.0, -img_t[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        H_t = cv2.invert(cv2.invert(H)[1] @ mat_t)[1]
        cornersD_t = cv2.perspectiveTransform(cornersD, H_t)

        print("MAPPING [Internal]: Warping image")
        output_img_size = (
            max(int(np.ceil(cornersD_t[:, :, 0].max())), wA) + img_ti[0],
            max(int(np.ceil(cornersD_t[:, :, 1].max())), hA) + img_ti[1],
        )
        img_delta_warped = cv2.warpPerspective(img_delta, H_t, output_img_size)

        mask = self.img_agg[:, :, 3] == 255
        img_delta_warped[img_ti[1] : img_ti[1] + hA, img_ti[0] : img_ti[0] + wA][
            mask
        ] = self.img_agg[mask]

        self.img_agg = img_delta_warped

        self.prev_kpD = kpD
        self.prev_desD = desD

        return True
