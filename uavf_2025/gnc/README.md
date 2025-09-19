# GNC 2025 Software

## Run the SITL
Installation instructions can be found [here](https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html)

1. In a new terminal source your build:
    ```
    source install/setup/bash
    ```

2. `cd` into your ros workspace, mine is located at `~/code/uavf25_ws`

3. Run **ONE** of these commands - Launch Simulation at ARC Mainfield West:
    ```
    sim_vehicle.py -v copter --console --map -w -l 33.642871,-117.826633,0.0,0.0
    ```
    Launch Simulation at ARC Club Field:
    ```
    sim_vehicle.py -v copter --console --map -w -l 33.642276,-117.827108,0.0,0.0
    ```
    Launch Simulation at ARC Main Field East:
    ```
    sim_vehicle.py -v copter --console --map -w -l 33.64199,-117.82497,0.0,0.0
    ```
    Launch Simulation at OCMA.
    ```
    sim_vehicle.py -v copter --console --map -w -l 33.7713180251761,-117.69482206241825,0.0,0.0
    ```
    Launch Simulation at Maryland.
    ```
    sim_vehicle.py -v copter --console --map -w -l 38.315386,-76.550875,0.0,0.0
    ```

4. Run this command if testing in simulation:
    ```
    output add 127.0.0.1:14550
    ```

- Note that if you were to use a different `.gpx` file you can change the home position by changing the args after `-l` to `lat,lon,alt,heading`

## Run MavROS
1. In a new terminal source your build:
    ```
    source install/setup/bash
    ```

2. `cd` into your ros workspace, mine is located at `~/code/uavf25_ws`

3. Run **ONE** of these commands (depending on wether your also using Qground):
    with Qground control:
    ```
    ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14540@localhost:14550 -p system_id:=255
    ```

    Without Qground:
    ```
    ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14550@localhost:14550 -p system_id:=255
    ```

## Run Commander Node
1. In a new terminal source your build:
    ```
    source install/setup/bash
    ```

2. `cd` into your ros workspace, mine is located at `~/code/uavf25_ws`

3. Run **ONE** of these commands (your preference):
   ```
   ros2 run uavf_2025 commander_node_demo.py src/uavf_2025/uavf_2025/gnc/data/main_field_west.gpx
   
   ros2 launch uavf_2025 commander_node.launch.py gpx_file:=src/uavf_2025/uavf_2025/gnc/data/main_field_west.gpx

   ```