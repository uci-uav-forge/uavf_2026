#!/bin/bash

# Run the geofence script and exit if it fails
# python3 send_geofence.py "$1"
echo "sent geofence. starting mavros..."

# Start mavros in the background
python3 start_mavros.py &

sleep 10
echo "starting commander node..."
# Run commander_node_demo in the foreground (outputs to console)
python3 commander_node_demo.py "$1"
