sudo apt install ros-humble-mavros ros-humble-mavros-extras
ros2 launch mavros px4.launch.py fcu_url:=/dev/ttyUSB0:57600
ros2 launch mavros px4.launch.py fcu_url:=udp://:14540@


# /image_raw in your VNS node.
# /mavros/local_position/pose with visual corrections.
# /mavros/setpoint_position/local to correct the drone’s position if needed.