./cleanup.sh
ros2 launch ardupilot_gz_bringup iris_runway.launch.py &
/home/$USER/Downloads/QGroundControl.AppImage > /dev/null 2>&1 &
sleep 10
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://127.0.0.1:14445@14445 -p system_id:=255 2>&1 | grep -vE 'mavros.param|mavros.distance_sensor' &
ros2 service call -r 3 /mavros/set_stream_rate mavros_msgs/srv/StreamRate "{stream_id: 0, message_rate: 15, on_off: true}" > /dev/null &
