import numpy as np
from scipy.spatial.transform import Rotation
from uavf_2025.shared.types import Pose


def test_interpolate_halfway():
    # Positions
    pos_a = np.array([0.0, 0.0, 0.0])
    pos_b = np.array([2.0, 2.0, 2.0])
    # Rotations
    rot_a = Rotation.from_euler("z", 0, degrees=True)
    rot_b = Rotation.from_euler("z", 90, degrees=True)
    # Create Poses
    pose_a = Pose(position=pos_a, rotation=rot_a)
    pose_b = Pose(position=pos_b, rotation=rot_b)
    # Interpolate at proportion=0.5
    proportion = 0.5
    interpolated_pose = pose_a.interpolate(pose_b, proportion)
    # Expected position
    expected_pos = np.array([1.0, 1.0, 1.0])
    # Expected rotation
    expected_rot = Rotation.from_euler("z", 45, degrees=True)
    # Check position
    assert np.allclose(
        interpolated_pose.position, expected_pos
    ), f"Interpolated position {interpolated_pose.position} does not match expected {expected_pos}"
    # Check rotation
    angle_diff = interpolated_pose.rotation.inv() * expected_rot
    angle_magnitude = np.linalg.norm(angle_diff.as_rotvec(), ord=2)
    assert (
        angle_magnitude < 1e-6
    ), f"Interpolated rotation differs from expected by {angle_magnitude} radians"


def test_extrapolate_before():
    # Positions
    pos_a = np.array([0.0, 0.0, 0.0])
    pos_b = np.array([2.0, 2.0, 2.0])
    # Rotations
    rot_a = Rotation.from_euler("z", 0, degrees=True)
    rot_b = Rotation.from_euler("z", 90, degrees=True)
    # Create Poses
    pose_a = Pose(position=pos_a, rotation=rot_a)
    pose_b = Pose(position=pos_b, rotation=rot_b)
    # Extrapolate at proportion=-1.0
    proportion = -1.0
    extrapolated_pose = pose_a.interpolate(pose_b, proportion)
    # Expected position
    expected_pos = np.array([-2.0, -2.0, -2.0])
    # Expected rotation
    expected_rot = Rotation.from_euler("z", -90, degrees=True)
    # Check position
    assert np.allclose(
        extrapolated_pose.position, expected_pos
    ), f"Extrapolated position {extrapolated_pose.position} does not match expected {expected_pos}"
    # Check rotation
    angle_diff = extrapolated_pose.rotation.inv() * expected_rot
    angle_magnitude = np.linalg.norm(angle_diff.as_rotvec(), ord=2)
    assert (
        angle_magnitude < 1e-6
    ), f"Extrapolated rotation differs from expected by {angle_magnitude} radians"


def test_extrapolate_after():
    # Positions
    pos_a = np.array([0.0, 0.0, 0.0])
    pos_b = np.array([2.0, 2.0, 2.0])
    # Rotations
    rot_a = Rotation.from_euler("z", 0, degrees=True)
    rot_b = Rotation.from_euler("z", 90, degrees=True)
    # Create Poses
    pose_a = Pose(position=pos_a, rotation=rot_a)
    pose_b = Pose(position=pos_b, rotation=rot_b)
    # Extrapolate at proportion=2.0
    proportion = 2.0
    extrapolated_pose = pose_a.interpolate(pose_b, proportion)
    # Expected position
    expected_pos = np.array([4.0, 4.0, 4.0])
    # Expected rotation
    expected_rot = Rotation.from_euler("z", 180, degrees=True)
    # Check position
    assert np.allclose(
        extrapolated_pose.position, expected_pos
    ), f"Extrapolated position {extrapolated_pose.position} does not match expected {expected_pos}"
    # Check rotation
    angle_diff = extrapolated_pose.rotation.inv() * expected_rot
    angle_magnitude = np.linalg.norm(angle_diff.as_rotvec(), ord=2)
    assert (
        angle_magnitude < 1e-6
    ), f"Extrapolated rotation differs from expected by {angle_magnitude} radians"


if __name__ == "__main__":
    test_interpolate_halfway()
    test_extrapolate_before()
    test_extrapolate_after()
    print("All tests passed!")
