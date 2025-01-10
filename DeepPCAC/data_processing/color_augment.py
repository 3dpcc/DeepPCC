# Camera Projection Related Functions
# Color Augment for Point Cloud
# Hao Zhu (zhuhao_nju@163.com)
# Last update: 2019-2-19

from __future__ import print_function
import numpy as np
import PIL.Image
import random
import cv2

class CamPara():
    def __init__(self, K=None, Rt=None):
        img_size = [200,200]
        if K is None:
            K = np.array([[500, 0, 112],
                          [0, 500, 112],
                          [0, 0, 1]])
        else:
            K = np.array(K)
        if Rt is None:
            Rt = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
        else:
            Rt = np.array(Rt)
        R = Rt[:,:3]
        t = Rt[:,3]

        self.cam_center = -np.dot(R.transpose(),t)
        # compute projection and inv-projection matrix
        self.proj_mat = np.dot(K, Rt)

        self.inv_proj_mat = np.linalg.pinv(self.proj_mat)

        # compute ray directions of camera center pixel
        c_uv = np.array([float(img_size[0])/2+0.5, float(img_size[1])/2+0.5])
        self.center_dir = self.inv_project(c_uv)

    def get_camcenter(self):
        return self.cam_center
    
    def get_center_dir(self):
        return self.center_dir
    
    def project(self, p_xyz):
        p_xyz = p_xyz.astype(np.double)
        p_uv_1 = np.dot(self.proj_mat, np.append(p_xyz, 1))
        if p_uv_1[2] == 0:
            return 0
        p_uv = (p_uv_1/p_uv_1[2])[:2]
        return p_uv
    
    # inverse projection, if depth is None, return a normalized direction
    def inv_project(self, p_uv, depth=None):
        p_uv = np.double(p_uv)
        p_xyz_1 = np.dot(self.inv_proj_mat, np.append(p_uv, 1))
        if p_xyz_1[3] == 0:
            return 0
        p_xyz = (p_xyz_1/p_xyz_1[3])[:3]
        p_dir = p_xyz - self.cam_center
        p_dir = p_dir / np.linalg.norm(p_dir)
        if depth is None:
            return p_dir
        else:
            real_xyz = self.cam_center + p_dir * depth
            return real_xyz


import PIL.Image
from io import BytesIO 
from IPython.display import display, Image
def show_img_arr(arr):
    im = PIL.Image.fromarray(arr)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))


def get_color_from_image(src_pc: object, src_img_file: object) -> object:
    K = np.array([[2000, 0, 250],
                [0, 2000, 250],
                [0, 0, 1]])
    # K = np.array([[1000, 0, 125],
    #             [0, 1000, 125],
    #             [0, 0, 1]])
    Rt = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -5]])
    cam_para = CamPara(K = K, Rt = Rt)
    # # read point cloud
    # src_pc = np.loadtxt("./point_cloud_test.txt")
    # src_pc = src_pc[:,:3].tolist()

    # read target texture image
    src_img = np.asarray(PIL.Image.open(src_img_file))
    img_size = src_img.shape
    if img_size[0] == img_size[1]:
        tgt_img = src_img
    elif img_size[0] > img_size[1]:
        start_pix = int((img_size[0] - img_size[1])*0.5)
        try:
            tgt_img = src_img[start_pix:start_pix+img_size[1], :, :]
        except IndexError: 
            tgt_img = src_img[:, start_pix:start_pix+img_size[0]]
            tgt_img = np.stack([tgt_img]*3, -1)
    else:
        start_pix = int((img_size[1] - img_size[0])*0.5)
        try:
            tgt_img = src_img[:, start_pix:start_pix+img_size[0], :]
            # IndexError: too many indices for array
        except IndexError: 
            tgt_img = src_img[:, start_pix:start_pix+img_size[0]]
            tgt_img = np.stack([tgt_img]*3, -1)
    # tgt_img = cv2.resize(tgt_img, (500, 500))
    tgt_img = tgt_img[(tgt_img.shape[0] - 500) //2 : 500 + (tgt_img.shape[0] - 500) // 2, (tgt_img.shape[1] - 500) //2 : 500 + (tgt_img.shape[1] - 500) // 2, :]
    print(tgt_img.shape)
    # get boundingbox sizetgt_img.shape[0]
    bdb_max = np.max(src_pc, axis = 0)
    bdb_min = np.min(src_pc, axis = 0)
    bdb_center = (bdb_max + bdb_min) * 0.5
    norm_scale = 1. / np.max(bdb_max - bdb_min)
    # transform pc
    trans_pc = (src_pc - bdb_center) * norm_scale

    # random rotate
    R = np.zeros((3, 3))
    posi_list = [0, 1, 2]
    random.shuffle(posi_list)
    R[0, posi_list[0]] = (random.randint(0, 1)-0.5)*2
    R[1, posi_list[1]] = (random.randint(0, 1)-0.5)*2
    R[2, posi_list[2]] = (random.randint(0, 1)-0.5)*2
    # get color
    vert_color = []
    for p in trans_pc:
        # p_r = np.dot(p, R)
        uv = cam_para.project(p)
        u = int(np.round(uv[0]))
        v = int(np.round(uv[1]))
        vert_color.append(tgt_img[u, v, :].tolist())
    return vert_color

