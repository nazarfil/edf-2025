

# After reboot, run the rest of this script manually or wrap everything below in a second script
# Add ROS 2 repo
sudo apt install software-properties-common -y
sudo add-apt-repository universe
sudo apt update

# Add ROS 2 GPG key
sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null

# Add ROS 2 source list
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# Install ROS 2 Humble desktop
sudo apt install ros-humble-desktop -y

# Source ROS 2 in bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install libcamera apps for Raspberry Pi camera v3
sudo apt install libcamera-apps -y
# (test manually with: libcamera-still -o test.jpg)

# Install camera-related ROS 2 packages
sudo apt install ros-humble-image-tools ros-humble-image-pipeline ros-humble-v4l2-camera -y

# Install colcon and build tools
sudo apt install python3-colcon-common-extensions python3-rosdep python3-argcomplete -y

# Initialize rosdep
sudo rosdep init
rosdep update

echo "ROS 2 + camera + colcon successfully installed."
echo "Reboot if you haven’t already, and create your workspace next:"
echo "mkdir -p ~/ros2_ws/src && cd ~/ros2_ws && colcon build"
