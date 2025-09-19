./partial_cleanup.sh
python3 spawn_dropzone_targets.py &
cd ../../..

python3 uavf_2025/gnc/dev/commander_node_demo.py uavf_2025/gnc/gcs/data/suas_runway_1.gpx --auto-arm

