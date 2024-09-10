#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        tan_fovx = np.tan(self.FoVx / 2.0)
        tan_fovy = np.tan(self.FoVy / 2.0)
        self.focal_y = self.image_height / (2.0 * tan_fovy)
        self.focal_x = self.image_width / (2.0 * tan_fovx)

    def print_camera_parameters(self):
        print(f"UID: {self.uid}")
        print(f"COLMAP ID: {self.colmap_id}")
        print(f"Rotation Matrix (R): \n{self.R}")
        print(f"Translation Vector (T): \n{self.T}")
        print(f"Field of View X (FoVx): {self.FoVx}")
        print(f"Field of View Y (FoVy): {self.FoVy}")
        print(f"Image Name: {self.image_name}")
        print(f"Data Device: {self.data_device}")
        print(f"Image Width: {self.image_width}")
        print(f"Image Height: {self.image_height}")
        print(f"Z-far: {self.zfar}")
        print(f"Z-near: {self.znear}")
        print(f"Translation: {self.trans}")
        print(f"Scale: {self.scale}")
        print(f"World View Transform: \n{self.world_view_transform}")
        print(f"Projection Matrix: \n{self.projection_matrix}")
        print(f"Full Projection Transform: \n{self.full_proj_transform}")
        print(f"Camera Center: {self.camera_center}")

    def point_in_frustum(self, point_3d):
        # 将3D点转换到相机坐标系
        P_w = np.array(point_3d)
        P_c = self.R @ (P_w - self.T)

        # 计算视锥体的范围
        tan_fovx = np.tan(self.FoVx / 2)
        tan_fovy = np.tan(self.FoVy / 2)

        # 判断点是否在视锥体内
        in_near_plane = P_c[2] <= -self.znear
        in_far_plane = P_c[2] >= -self.zfar
        in_left_plane = P_c[0] >= -P_c[2] * tan_fovx
        in_right_plane = P_c[0] <= P_c[2] * tan_fovx
        in_top_plane = P_c[1] <= P_c[2] * tan_fovy
        in_bottom_plane = P_c[1] >= -P_c[2] * tan_fovy

        in_frustum = in_near_plane and in_far_plane and in_left_plane and in_right_plane and in_top_plane and in_bottom_plane
        return in_frustum
    
    def pixel_footprint(self,point_3d):
        # 物體在世界坐標下的三維座標
        P_w = np.array(point_3d)

        # 相機的旋轉矩陣
        R = np.array(self.R)

        # 相機的平移向量
        T = np.array(self.T)

        # 計算相機坐標系中的點 P_c
        P_c = R.dot(P_w - T)

        # 焦距（假設相機內部參數）
        f = np.array(self.projection_matrix.cpu().numpy())[0][0]

        # 計算視野範圍內的條件
        '''
        in_view_x = abs(P_c[0] / P_c[2]) < self.FoVx #np.tan(self.FoVx / 2)
        in_view_y = abs(P_c[1] / P_c[2]) < self.FoVy #np.tan(self.FoVy / 2)
        in_view = in_view_x and in_view_y'''
        in_view = self.point_in_frustum(point_3d)

        # 計算相機到物體的距離
        d = np.linalg.norm(P_w - T)
        
        print(in_view)
        # 算出了P_c, in_view, d
        if in_view:
            return d/f
        else:
            return 800

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

'''
    def is_is_visible(self,point):
# 将三维坐标转换为齐次坐标
        point_3d_homogeneous = torch.tensor([point[0], point[1], point[2], 1], dtype=torch.float32, device=self.device)

        # 应用投影矩阵
        projected_point_homogeneous = torch.matmul(self.projection_matrix, point_3d_homogeneous)

        # 归一化齐次坐标，得到二维图像平面坐标
        x_img = projected_point_homogeneous[0] / projected_point_homogeneous[3]
        y_img = projected_point_homogeneous[1] / projected_point_homogeneous[3]

        # 将张量转换为 Python 浮点数
        x_img_float = x_img.item()
        y_img_float = y_img.item()

        projected_point = [x_img_float, y_img_float]
        print("===============is_is_projected_point: ", projected_point)

        # 定义图像边界为多边形
        image_polygon = np.array([[0, 0], [self.image_width, 0], [self.image_width, self.image_height], [0, self.image_height]], dtype=np.int32)

        # 检查投影点是否在图像边界内
        is_inside = cv2.pointPolygonTest(image_polygon, (projected_point[0], projected_point[1]), False)
        print(is_inside)

        # 如果点在图像内，则返回True，否则返回False
        return is_inside >= 0

    def is_visible(self, point):
        point = np.array(point)
        # 计算内参矩阵
        self.fx = self.FoVx
        self.fy = self.FoVy
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        print("===========ID+++++++++++++: ",self.uid)
        print("==============FoVX============:",self.FoVx)
        print("==============FoVX============:",self.FoVx)
        self.intrinsic_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        rvec = self.R
        tvec = self.T

        dist_coeffs = np.zeros(5, dtype=np.float32) # suppose there's no distortion
        image_points, _ = cv2.projectPoints(objectPoints = point, rvec = rvec, tvec = tvec,
                                             cameraMatrix = self.intrinsic_matrix, distCoeffs = dist_coeffs)
        projected_point = image_points[0][0]
        print("==============projected============:",projected_point)
        # 将张量转换为普通的 Python 浮点数
        projected_point_x = projected_point[0].item()
        projected_point_y = projected_point[1].item()
        # Define the image boundaries as a polygon
        image_polygon = np.array([[0, 0], [self.image_width, 0], [self.image_width, self.image_height], [0, self.image_height]], dtype=np.int32)

        # Check if the projected point is inside the image boundaries
        is_inside = cv2.pointPolygonTest(image_polygon, (projected_point_x, projected_point_y), False)
        #print(is_inside)
        # Return True if the point is inside the image, otherwise False
        return is_inside >= 0
    


    def calculate_pixel_footprint(self, point_3d):
        print('sdfja;qerohnowjgfawefjawefj:      ')
        print(point_3d)
        if self.is_visible(point_3d):
            point_3d = np.array(point_3d)

            # Convert 3D point to camera coordinate system
            point_cam = self.R @ (point_3d - self.T) 

            # Compute the focal lengths
            fx = self.FoVx
            fy = self.FoVy

            # Compute pixel footprint in world space
            z = point_cam[2]
            pixel_footprint_x = z / fx
            pixel_footprint_y = z / fy
            min = -100
            if pixel_footprint_x>0:
                min = pixel_footprint_x
            if pixel_footprint_y < pixel_footprint_x:
                min = pixel_footprint_y

            return min
'''