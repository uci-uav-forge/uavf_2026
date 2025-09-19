from typing import Callable, ParamSpec, TypeVar

import rclpy

Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")


def ensure_ros(func: Callable[Params, ReturnType]) -> Callable[Params, ReturnType]:
    """
    Decorator for a function to start rclpy if it has not been started yet.
    """

    def wrapper(*args, **kwargs):
        if not rclpy.ok():
            rclpy.init()

        return func(*args, **kwargs)

    return wrapper
