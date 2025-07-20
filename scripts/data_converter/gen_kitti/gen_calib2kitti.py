import os
import numpy as np
from .utils import mkdir_p, read_json, get_files_path


def convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam):
    P2 = np.zeros([3, 4]) # 创建一个 3x4 的零矩阵
    P2[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")#cam_K 被用来填充这个矩阵的前 3×3 部分
    P2 = P2.reshape(12, order="C")

    Tr_velo_to_cam = np.concatenate((r_velo2cam, t_velo2cam), axis=1)
    Tr_velo_to_cam = Tr_velo_to_cam.reshape(12, order="C")

    return P2, Tr_velo_to_cam


def get_cam_D_and_cam_K(path):
    my_json = read_json(path)
    cam_D = my_json["cam_D"]
    cam_K = my_json["cam_K"]
    return cam_D, cam_K


def get_velo2cam(path):
    my_json = read_json(path)
    t_velo2cam = my_json["translation"]
    r_velo2cam = my_json["rotation"]
    return t_velo2cam, r_velo2cam


def gen_calib2kitti(path_camera_intrisinc, path_lidar_to_camera, path_calib):
    path_list_camera_intrisinc = get_files_path(path_camera_intrisinc, ".json")
    path_list_lidar_to_camera = get_files_path(path_lidar_to_camera, ".json")
    path_list_camera_intrisinc.sort()
    path_list_lidar_to_camera.sort()
    #print(len(path_list_camera_intrisinc), len(path_list_lidar_to_camera))
    print(f"path_list_camera_intrisinc,相机内参:{len(path_list_camera_intrisinc)}")
    print(f"path_list_lidar_to_camera,雷达To相机:{len(path_list_lidar_to_camera)}")
    mkdir_p(path_calib)

    for i in range(len(path_list_camera_intrisinc)):
        cam_D, cam_K = get_cam_D_and_cam_K(path_list_camera_intrisinc[i])#提取相机畸变系数 cam_D 和相机矩阵 cam_K
        t_velo2cam, r_velo2cam = get_velo2cam(path_list_lidar_to_camera[i])#从 LiDAR 到相机的变换中提取平移向量 t_velo2cam 和旋转矩阵 r_velo2cam
        json_name = os.path.split(path_list_camera_intrisinc[i])[-1][:-5] + ".txt"#.json 改成 .txt，作为输出文件名
        json_path = os.path.join(path_calib, json_name)

        t_velo2cam = np.array(t_velo2cam).reshape(3, 1)
        r_velo2cam = np.array(r_velo2cam).reshape(3, 3)
        P2, Tr_velo_to_cam = convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam)

        str_P2 = "P2: "
        str_Tr_velo_to_cam = "Tr_velo_to_cam: "
        for ii in range(11):
            str_P2 = str_P2 + str(P2[ii]) + " "
            str_Tr_velo_to_cam = str_Tr_velo_to_cam + str(Tr_velo_to_cam[ii]) + " "
        str_P2 = str_P2 + str(P2[11])
        str_Tr_velo_to_cam = str_Tr_velo_to_cam + str(Tr_velo_to_cam[11])

        str_P0 = str_P2
        str_P1 = str_P2
        str_P3 = str_P2
        str_R0_rect = "R0_rect: 1 0 0 0 1 0 0 0 1"
        str_Tr_imu_to_velo = str_Tr_velo_to_cam

        with open(json_path, "w") as fp:
            gt_line = (
                str_P0
                + "\n"
                + str_P1
                + "\n"
                + str_P2
                + "\n"
                + str_P3
                + "\n"
                + str_R0_rect
                + "\n"
                + str_Tr_velo_to_cam
                + "\n"
                + str_Tr_imu_to_velo
            )
            fp.write(gt_line)
