# Perception Development Guidelines

## Testing Code
Two common ways to test your code with the rest of the drone system:
1. rosbag (simplest)
2. QGroundControl/Gazebo (more rigorous)

### Using rosbag
1. Get a rosbag *folder* `rosbag2_*/` containing `metadata.yaml` and `rosbag2_*.db3`
2. Run perception main
    - e.g., `python uavf_2025/perception/dev/perception_cli.py sim`
    - make sure you sourced your venv or started your conda (if any)
3. Run `ros2 bag play */rosbag2_* --loop`

You can also generate rosbags from our logs. Refer `logs/to_rosbag.py`

### Using QGroundControl / Gazebo
Run 4 separate commands **simultaneously**. Here are some *example* commands:

1. `alias uavf_mavros_qgc='ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://127.0.0.1:14445@14445 -p system_id:=255'`
2. `alias uavf_runway="ros2 launch ardupilot_gz_bringup iris_runway.launch.py"`
    - **Run `uavf_python` before**
3. `alias uavf_qground="~/QGroundControl.AppImage`
    - Download the QGroundControl first
4. `alias uavf_python="cd Documents/ardu_ws/src/uavf_2025; source venv/bin/activate"` and run perception main

Additional tips:
You can group the first three commands together as an alias:

`alias uavf_system="(trap 'echo \"Killing background processes...\"; kill 0; exit' SIGINT; uavf_mavros_qgc & (uavf_python; uavf_runway) & uavf_qground & wait)"`

And run the perception main separately.

