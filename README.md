# UAV Forge Flight Software 2025

## Setup
1. Clone the repo into the src directory of a ROS workspace. For example, Eric's is in `~/code/uavf_ws/src/uavf_2025`. The ROS workspace part is the `uavf_ws` folder. This can be called whatever, but the `src` directory needs to be called `src`.

2. cd into this folder and run these commands:
   ```
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    pre-commit install
   ```

## Building and running the main ROS node
1. `cd` into your ROS ws directory. (ex: `~/code/uavf_ws`)
2. `colcon build` (if there are other packages, you can just do `colcon build --packages-select uavf_2025`)
3. `source install/setup.bash`
4. `ros2 launch uavf_2025 main_node.launch.py`

## Running unit tests
1. `cd` into the directory the repository is cloned into (uavf_2025)
2. run this command: `pytest`

## Handy dandy ROS commands
remember to `source install/setup.bash` and `colcon build` from the ws directory
Spawning stuff: `ros2 run ros_gz_sim create -file /home/miller/code/ardu_ws/src/ardupilot_gazebo/models/stop_sign -name target_4 -x 0 -y 0 -z 0 --ros-args --log-level error`

`ros2 launch ardupilot_gz_bringup iris_runway.launch.py`
`ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://127.0.0.1:14550@14550 -p system_id:=255`

## Payload
To set up a real payload module on a new device add these lines to your `~/.bashrc` or enviorment
```bash
export PAYLOAD_SERIAL_PORT = "your_serial_path"
```

To find your serial path use the following command to view the latest plugged in port:
```bash
sudo dmesg
```

<br>

To drop a gazebo payload manually run the following:
```bash
 ros2 service call uavf_drop_srv std_srvs/srv/Empty
```