import rclpy


def call_with_timeout(node, client, request, timeout_sec):
    return client.call(request)
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    return future.result()
