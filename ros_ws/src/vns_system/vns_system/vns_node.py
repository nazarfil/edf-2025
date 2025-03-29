#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import random

class VNSNode(Node):
    def __init__(self):
        super().__init__('vns_node')
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/vns/position_update', 10)
        self.timer = self.create_timer(30.0, self.publish_mock_position)  # Every 30 seconds
        self.get_logger().info("VNS node started.")

    def image_callback(self, msg):
        self.get_logger().info('Received camera frame')
        # Here you would normally process the image

    def publish_mock_position(self):
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        # Mock position: random drift around a base point
        base_lat, base_lon = 50.8503, 4.3517  # Brussels GPS coords (for example)
        pose_msg.pose.position.x = base_lat + random.uniform(-0.0005, 0.0005)
        pose_msg.pose.position.y = base_lon + random.uniform(-0.0005, 0.0005)
        pose_msg.pose.position.z = 100.0  # Constant altitude

        self.pub.publish(pose_msg)
        self.get_logger().info(f"Published position: {pose_msg.pose.position}")

def main(args=None):
    rclpy.init(args=args)
    node = VNSNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
