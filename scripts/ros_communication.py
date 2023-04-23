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

global flag_camera_info
flag_camera_info = True
# TODO: unified pose and image at the same time: depth main
# TODO: unified scene and depth at the same time: seq
global xform
xform = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,1.0,0.0,0.0],
                  [0.0,0.0,1.0,0.0],
                  [0.0,0.0,0.0,1.0]])
global output_dir
output_dir = opj(
    opd(opd(__file__)),
    "output"
)
# test
global target_seq_scene
target_seq_scene = 0
global target_seq_depth
target_seq_depth = 0
global idx_scene
idx_scene = -1
global idx_depth
idx_depth = -1
global interval
interval = 5
# GPU
global max_image
max_image = 5
# end
global target_end_seq
target_end_seq = 50
global old_xform
old_xform = np.array([[1.0,0.0,0.0,0.0],
                  [0.0,1.0,0.0,0.0],
                  [0.0,0.0,1.0,0.0],
                  [0.0,0.0,0.0,1.0]])

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Ros_Node:
    def __init__(self, testbed: ngp.Testbed) -> None:
        self.testbed = testbed
        self.clear_json()
        # rospy.Subscriber(f"/msg", String, self.msg_recv_cb)
        # rospy.Subscriber(f"/pos", PoseStamped, self.pos_recv_cb)
        # rospy.Subscriber(f"/camera_info", CameraInfo, self.camera_info_recv_cb)
        # rospy.Subscriber(f"/image", Image, self.image_recv_cb)
        drone_id = 1
        rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera_vis/DepthVis/camera_info", CameraInfo, self.camera_info_recv_cb)
        rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera/pose", PoseStamped, self.pos_recv_cb)
        rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_scene_camera/Scene", Image, self.image_scene_recv_cb)
        rospy.Subscriber(f"/airsim_node/uav{drone_id}/uav{drone_id}_depth_camera_vis/DepthVis", Image, self.image_depth_recv_cb)

    def clear_json(self):
        param = {
            'camera_angle_x': 0.0,
            'camera_angle_y': 0.0,
            'aabb_scale': 1,
            'scale': 0.8,
            'enable_depth_loading': True,
            'depth_range_realunit': 100,
            'real2ngp_uint_conersion': 0.025,
            'depth_bit_depth': 8,
            'n_extra_learnable_dims': 0
        }
        data_json = json.dumps(param, cls=NumpyArrayEncoder, indent=2)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = opj(
            output_dir,
            "test.json"
        )
        with open(file_name, "w", newline='\n') as f:
            f.write(data_json)

    def msg_recv_cb(self, message: String):
        print(f'message recved: {message.data}')

    def pos_recv_cb(self, pos: PoseStamped):
        print(f"pos stamp: {pos.header.stamp}")
        # print(f"pos seq: {pos.header.seq}")
        # return
        global xform
        global old_xform
        old_xform = xform
        # TODO: roi confirm
        # roi_center_meter = [8,3,-4]
        # roi_len_of_side_meter = 40
        # roi_center_meter = [1,6,2]
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
        # print(f"old_xform:{old_xform}")
        # print(f"xform:{xform}")
        # param = {
        #     'transform_matrix': xform
        # }

        # with open("/home/mumu/uav-nerf-test/output/test.json", "r") as f:
        #     content = json.load(f)
        # content["frames"].append(param)
        # data_json = json.dumps(content, cls=NumpyArrayEncoder, indent=2)
        # with open("/home/mumu/uav-nerf-test/output/test.json", "w", newline='\n') as f:
        #     f.write(data_json)
        #     print("recved")

        # self.testbed.reload_training_data()
        # TODO: init again?

    def camera_info_recv_cb(self, camera_info: CameraInfo):
        # TODO: camera_angle should come from camsetting
        # TODO: fov_degree should come from camsetting
        # TODO: depth_range_realunit should come from camsetting
        # TODO: depth_bit_depth should come from camsetting

        # only use once
        global flag_camera_info 
        global output_dir
        if (flag_camera_info):
            if(len(camera_info.D) == 0):
                param = {
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
                    'frames': []
                }
            else:
                param = {
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
                    'frames': []
                }

            file_name = opj(
                output_dir,
                "test.json"
            )
            with open(file_name, "r") as f:
                content = json.load(f)
            content.update(param)
            data_json = json.dumps(content, indent=2)
            with open(file_name, "w", newline='\n') as f:
                f.write(data_json)
            flag_camera_info = False

    def image_scene_recv_cb(self, image: Image):
        print(f"scene stamp:{image.header.stamp}")
        # print(f"scene seq:{image.header.seq}")
        # return
        global output_dir
        global target_seq_scene
        global idx_scene
        global idx_depth
        global interval
        global xform
        global old_xform
        global flag_camera_info
        global max_image
        global target_end_seq
        if (flag_camera_info):
            return
        # if (np.array_equal(old_xform, xform)):
        #     return
        seq = image.header.seq
        if (seq > target_end_seq):
            return
        if (seq > target_seq_scene):
            target_seq_scene += interval
        elif (seq == target_seq_scene):
            img = CvBridge().imgmsg_to_cv2(image)
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
            # filename = f"/home/mumu/uav-nerf-test/output/image/{seq}.png"
            cv2.imwrite(filename, img)   

            param = {
                'file_path': f"image/{seq}.png",
                'transform_matrix': xform,
                'seq': seq
            }

            file_name = opj(
                output_dir,
                "test.json"
            )
            update_flag_scene = False
            if(idx_scene >= max_image):
                update_flag_scene = True
            flag_reload = False
            with open(file_name, "r") as f:
                content = json.load(f)
            if (idx_depth == idx_scene):
                if (update_flag_scene):
                    idx = (idx_scene + 1) % (max_image + 1)
                    content["frames"][idx].update(param)
                else:
                    content["frames"].append(param)
            elif (idx_scene == idx_depth - 1):
                if (update_flag_scene):
                    idx = idx_depth % (max_image + 1)
                    target_seq = content["frames"][idx]["seq"]
                    content["frames"][idx].update(param)
                    if (seq != target_seq):
                        print(f"seq:{seq}")
                        print(f"target_seq:{target_seq}")
                        idx_depth -= 1
                    else:
                        flag_reload = True
                else:
                    target_seq = content["frames"][idx_depth]["seq"]
                    content["frames"][idx_depth].update(param)
                    if (seq != target_seq):
                        print(f"seq:{seq}")
                        print(f"target_seq:{target_seq}")
                        idx_depth -= 1
                    else:
                        flag_reload = True
            idx_scene += 1
            data_json = json.dumps(content, cls=NumpyArrayEncoder, indent=2)
            with open(file_name, "w", newline='\n') as f:
                f.write(data_json) 
                print(f"scene seq:{image.header.seq}")
                print("recved")  
            if (flag_reload):
                self.testbed.reload_training_data()
                # self.testbed.training_step = 0

    def image_depth_recv_cb(self, image: Image):
        print(f"depth stamp:{image.header.stamp}")
        # print(f"depth seq:{image.header.seq}")
        # return
        global output_dir
        global target_seq_depth
        global idx_scene
        global idx_depth
        global interval
        global xform
        global old_xform
        global flag_camera_info
        global max_image
        global target_end_seq
        if (flag_camera_info):
            return
        # if (np.array_equal(old_xform, xform)):
        #     return
        seq = image.header.seq
        if (seq > target_end_seq):
            return
        if (seq > target_seq_depth):
            target_seq_depth += interval
        elif (seq == target_seq_depth):
            img = CvBridge().imgmsg_to_cv2(image)
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
            # filename = "/home/mumu/uav-nerf-test/output/depth/{seq}.png"
            cv2.imwrite(filename, img)   

            param = {
                'depth_path': f"depth/{seq}.png",
                'transform_matrix': xform,
                'seq': seq
            }

            file_name = opj(
                output_dir,
                "test.json"
            )
            update_flag_depth = False
            if(idx_depth >= max_image):
                update_flag_depth = True
            flag_reload = False
            with open(file_name, "r") as f:
                content = json.load(f)
            if (idx_depth == idx_scene):
                if (update_flag_depth):
                    idx = (idx_depth + 1) % (max_image + 1)
                    content["frames"][idx].update(param)
                else:
                    content["frames"].append(param)
            elif (idx_depth == idx_scene - 1):
                if (update_flag_depth):
                    idx = idx_scene % (max_image + 1)
                    target_seq = content["frames"][idx]["seq"]
                    content["frames"][idx].update(param)
                    if (seq != target_seq):
                        print(f"seq:{seq}")
                        print(f"target_seq:{target_seq}")
                        idx_scene -= 1
                    else:
                        flag_reload = True
                else:
                    target_seq = content["frames"][idx_scene]["seq"]
                    content["frames"][idx_scene].update(param)
                    if (seq != target_seq):
                        print(f"seq:{seq}")
                        print(f"target_seq:{target_seq}")
                        idx_scene -= 1
                    else:
                        flag_reload = True
            idx_depth += 1
            data_json = json.dumps(content, cls=NumpyArrayEncoder, indent=2)
            with open(file_name, "w", newline='\n') as f:
                f.write(data_json)
                print(f"depth seq:{image.header.seq}") 
                print("recved")   
            if (flag_reload):
                self.testbed.reload_training_data() 
                # self.testbed.training_step = 0
        

def thread_job():
    rospy.spin()

def listener(testbed: ngp.Testbed):
    rospy.init_node('ros_node', anonymous=True)
    rate = rospy.Rate(10)

    ros_node = Ros_Node(testbed)

    spin_thread = threading.Thread(target = thread_job)
    spin_thread.start()