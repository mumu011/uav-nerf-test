import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import threading
import numpy as np
import json
from cv_bridge import CvBridge
import cv2
import os
from os.path import join as opj
from os.path import dirname as opd
import message_filters

import pyngp as ngp
from transform import transform

global output_dir
output_dir = opj(
    opd(opd(__file__)),
    "output_v2"
)
# test
global target_seq
target_seq = 0
global interval
interval = 1
# GPU
global max_image
max_image = 50
global json_idx
json_idx = 0
# end
global target_end_seq
target_end_seq = 100
# TODO: unified pose and image at the same time: depth main
# TODO: unified scene and depth at the same time: seq
global position
position = PoseStamped()
global position2
position2 = PoseStamped()
# test
global pos_stamp
pos_stamp = 0

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Ros_Node:
    def __init__(self, testbed: ngp.Testbed) -> None:
        self.testbed = testbed
        self.clear_json(json_idx)
        # rospy.Subscriber(f"/msg", String, self.msg_recv_cb)
        # rospy.Subscriber(f"/pos", PoseStamped, self.pos_recv_cb)
        # rospy.Subscriber(f"/camera_info", CameraInfo, self.camera_info_recv_cb)
        # rospy.Subscriber(f"/image", Image, self.image_recv_cb)
        drone_id = 1
        # rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera_vis/DepthVis/camera_info", CameraInfo, self.camera_info_recv_cb)
        rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera/pose", PoseStamped, self.pos_recv_cb, drone_id)
        # rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_scene_camera/Scene", Image, self.image_scene_recv_cb)
        # rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera_vis/DepthVis", Image, self.image_depth_recv_cb)

        info_sub = message_filters.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera_vis/DepthVis/camera_info", CameraInfo)
        # pos_sub = message_filters.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera/pose", PoseStamped)
        scene_sub = message_filters.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_scene_camera/Scene", Image)
        depth_sub = message_filters.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera_vis/DepthVis", Image)

        # ts = message_filters.ApproximateTimeSynchronizer([info_sub, pos_sub, scene_sub, depth_sub], 1000, 0.001)
        ts = message_filters.ApproximateTimeSynchronizer([info_sub, scene_sub, depth_sub], 100, 1)
        # ts.registerCallback(self.recv_callback)
        ts.registerCallback(self.recv_test_callback, drone_id)

        # multiple
        drone_id_two = 2
        rospy.Subscriber(f"/airsim_node/uav{drone_id_two}/uav{drone_id_two}_depth_camera/pose", PoseStamped, self.pos_recv_cb, drone_id_two)
        info_sub_two = message_filters.Subscriber(f"/airsim_node/uav{drone_id_two}/uav{drone_id_two}_depth_camera_vis/DepthVis/camera_info", CameraInfo)
        scene_sub_two = message_filters.Subscriber(f"/airsim_node/uav{drone_id_two}/uav{drone_id_two}_scene_camera/Scene", Image)
        depth_sub_two = message_filters.Subscriber(f"/airsim_node/uav{drone_id_two}/uav{drone_id_two}_depth_camera_vis/DepthVis", Image)

        ts_two = message_filters.ApproximateTimeSynchronizer([info_sub_two, scene_sub_two, depth_sub_two], 100, 1)
        ts_two.registerCallback(self.recv_test_callback, drone_id_two)

    def pos_recv_cb(self, pos: PoseStamped, drone_id: int):
        print(f"drone_id {drone_id} pos stamp: {pos.header.stamp.to_sec()}")
        # print(f"pos seq: {pos.header.seq}")
        # return
        global position
        global position2
        global pos_stamp
        if (drone_id == 1):
            position = pos
        elif (drone_id == 2):
            position2 = pos
        # pos_stamp = pos.header.stamp.to_sec()    

    def recv_test_callback(self, camera_info: CameraInfo, scene: Image, depth: Image, drone_id: int):
        global position
        global position2
        global pos_stamp
        global target_seq_scene
        global target_seq
        global interval
        global max_image
        global json_idx
        # print(f"drone_id:{drone_id}")
        seq_scene = scene.header.seq
        seq_depth = depth.header.seq
        # test
        scene_stamp = scene.header.stamp.to_sec()
        depth_stamp = depth.header.stamp.to_sec()
        if (seq_scene != seq_depth):
            return
        print(f"drone_id {drone_id} seq_scene test:{seq_scene}")
        print(f"drone_id {drone_id} seq_depth test:{seq_depth}")
        # if (seq_scene > target_end_seq):
        #     return
        if (seq_scene > target_seq and interval != 1):
            target_seq += interval
        elif (seq_scene == target_seq or interval == 1):
            if(len(camera_info.D) == 0):
                param_info = {
                    'fl_x': camera_info.K[0],
                    'fl_y': camera_info.K[4],
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0,
                    'cx': camera_info.K[2],
                    'cy': camera_info.K[5],
                    'w': camera_info.width,
                    'h': camera_info.height,
                }
            else:
                param_info = {
                    'fl_x': camera_info.K[0],
                    'fl_y': camera_info.K[4],
                    'k1': camera_info.D[0],
                    'k2': camera_info.D[1],
                    'p1': camera_info.D[2],
                    'p2': camera_info.D[3],
                    'cx': camera_info.K[2],
                    'cy': camera_info.K[5],
                    'w': camera_info.width,
                    'h': camera_info.height,
                }

            img_scene = CvBridge().imgmsg_to_cv2(scene)
            image_dir = opj(
                output_dir,
                "image"
            )
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            filename = opj(
                image_dir,
                f"{drone_id}_{seq_scene}.png"
            )
            cv2.imwrite(filename, img_scene)

            img_depth = CvBridge().imgmsg_to_cv2(depth)
            image_dir = opj(
                output_dir,
                "depth"
            )
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            filename = opj(
                image_dir,
                f"{drone_id}_{seq_depth}.png"
            )
            cv2.imwrite(filename, img_depth)   

            file_name = opj(
                output_dir,
                f"test_{json_idx}.json"
            )
            with open(file_name, "r") as f:
                content = json.load(f)
            content.update(param_info)
            if (drone_id == 1):
                quaternion = position.pose.orientation
                position_meter = position.pose.position
            elif (drone_id == 2):
                quaternion = position2.pose.orientation
                position_meter = position2.pose.position
            param_img = {
                'file_path': f"image/{drone_id}_{seq_scene}.png",
                'depth_path': f"depth/{drone_id}_{seq_depth}.png",
                'quaternion': {
                    'w': quaternion.w,
                    'x': quaternion.x,
                    'y': quaternion.y,
                    'z': quaternion.z,
                },
                'position': {
                    'x': position_meter.x,
                    'y': position_meter.y,
                    'z': position_meter.z,
                },
                # 'scene_stamp': scene_stamp,
                # 'depth_stamp': depth_stamp,
                # 'pos_stamp': pos_stamp
            }
            content["frames"].append(param_img)
            img_num = len(content["frames"])
            data_json = json.dumps(content, indent=2)
            with open(file_name, "w", newline='\n') as f:
                f.write(data_json)

            if (img_num % max_image == 0):
                json_idx += 1
                self.clear_json(json_idx)
        

    def recv_callback(self, camera_info: CameraInfo, pos: PoseStamped, scene: Image, depth: Image):
        seq = scene.header.seq
        print(f"seq:{seq}")
        if (seq > target_seq):
            target_seq += interval
        elif (seq == target_seq):
            # 仿真的中心
            roi_center_meter = [1,1,1]
            roi_len_of_side_meter = 40
            grid_levels = 1
            unit_per_meter = 1.0/roi_len_of_side_meter*(1<<(grid_levels-1));
            # compute xform
            quaternion = pos.pose.orientation
            position_meter = pos.pose.position
            rotation_matrix = np.array([[1.0-2.0*quaternion.y*quaternion.y-2.0*quaternion.z*quaternion.z,2.0*quaternion.x*quaternion.y-2.0*quaternion.w*quaternion.z,2.0*quaternion.z*quaternion.x+2.0*quaternion.w*quaternion.y,0.0],
                                        [2.0*quaternion.x*quaternion.y+2.0*quaternion.w*quaternion.z,1.0-2.0*quaternion.x*quaternion.x-2.0*quaternion.z*quaternion.z,2.0*quaternion.y*quaternion.z-2.0*quaternion.w*quaternion.x,0.0],
                                        [2.0*quaternion.z*quaternion.x-2.0*quaternion.w*quaternion.y,2.0*quaternion.y*quaternion.z+2.0*quaternion.w*quaternion.x,1.0-2.0*quaternion.x*quaternion.x-2.0*quaternion.y*quaternion.y,0.0],
                                        [0.0                                                        ,0.0                                                        ,0.0                                                            ,1.0]])
            # print(rotation_matrix)
            rotation_matrix = rotation_matrix[:,[1,2,0,3]]
            # rotation_matrix = rotation_matrix[:,[2,1,0,3]]
            # print(rotation_matrix)
            # TODO: changes
            translation_matrix = np.array([[1.0,0.0,0.0,(position_meter.x-roi_center_meter[0])*unit_per_meter],
                                        [0.0,1.0,0.0,(position_meter.y-roi_center_meter[1])*unit_per_meter],
                                        [0.0,0.0,1.0,(position_meter.z-roi_center_meter[2])*unit_per_meter],
                                        [0.0,0.0,0.0,1.0]])
            xform = np.dot(translation_matrix, rotation_matrix)

            if(len(camera_info.D) == 0):
                param_info = {
                    'fl_x': camera_info.K[0],
                    'fl_y': camera_info.K[4],
                    'k1': 0.0,
                    'k2': 0.0,
                    'p1': 0.0,
                    'p2': 0.0,
                    'cx': camera_info.K[2],
                    'cy': camera_info.K[5],
                    'w': camera_info.width,
                    'h': camera_info.height,
                }
            else:
                param_info = {
                    'fl_x': camera_info.K[0],
                    'fl_y': camera_info.K[4],
                    'k1': camera_info.D[0],
                    'k2': camera_info.D[1],
                    'p1': camera_info.D[2],
                    'p2': camera_info.D[3],
                    'cx': camera_info.K[2],
                    'cy': camera_info.K[5],
                    'w': camera_info.width,
                    'h': camera_info.height,
                }

            img_scene = CvBridge().imgmsg_to_cv2(scene)
            image_dir = opj(
                output_dir,
                "image"
            )
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            filename = opj(
                image_dir,
                f"{seq}.png"
            )
            cv2.imwrite(filename, img_scene)

            img_depth = CvBridge().imgmsg_to_cv2(depth)
            image_dir = opj(
                output_dir,
                "depth"
            )
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            filename = opj(
                image_dir,
                f"{seq}.png"
            )
            cv2.imwrite(filename, img_depth)   


            file_name = opj(
                output_dir,
                "test.json"
            )
            with open(file_name, "r") as f:
                content = json.load(f)
            content.update(param_info)
            param_img = {
                'file_path': f"image/{seq}.png",
                'depth_path': f"depth/{seq}.png",
                'transform_matrix': xform
            }
            content["frames"].append(param_img)
            data_json = json.dumps(content, indent=2)
            with open(file_name, "w", newline='\n') as f:
                f.write(data_json)

            # self.testbed.reload_training_data() 
            # self.testbed.training_step = 0

    def clear_json(self, json_idx: int):
        param = {
            'camera_angle_x': 0.0,
            'camera_angle_y': 0.0,
            'aabb_scale': 1,
            'scale': 0.8,
            'enable_depth_loading': True,
            'depth_range_realunit': 100,
            'real2ngp_uint_conersion': 0.025,
            'depth_bit_depth': 8,
            'n_extra_learnable_dims': 0,
            'frames': []
        }
        data_json = json.dumps(param, cls=NumpyArrayEncoder, indent=2)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = opj(
            output_dir,
            f"test_{json_idx}.json"
        )
        with open(file_name, "w", newline='\n') as f:
            f.write(data_json)

    def msg_recv_cb(self, message: String):
        print(f'message recved: {message.data}')
        

def thread_job():
    rospy.spin()

def listener(testbed: ngp.Testbed):
    rospy.init_node('ros_node', anonymous=True)
    rate = rospy.Rate(10)

    ros_node = Ros_Node(testbed)

    spin_thread = threading.Thread(target = thread_job)
    spin_thread.start()