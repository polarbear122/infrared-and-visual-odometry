from __future__ import print_function
import urllib.request as ur
import bz2
import os
import numpy as np
import sys
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
'''
a1 = np.array([[1, 2, 3, 4]])
print(a1)
print("数据类型", type(a1))  # 打印数组数据类型
print("数组元素数据类型：", a1.dtype)  # 打印数组元素数据类型
print("数组元素总数：", a1.size)  # 打印数组尺寸，即数组元素总数
print("数组形状：", a1.shape)  # 打印数组形状
print("数组的维度数目", a1.ndim)  # 打印数组的维度数目
'''
#First download the data file:
#BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
#FILE_NAME = "problem-49-7776-pre.txt.bz2"
#URL = BASE_URL + FILE_NAME
#if not os.path.isfile(FILE_NAME):
#    ur.urlretrieve(URL, FILE_NAME)
'''
数据格式
相机数目    点的数目    观察数目
相机序号    点的序号    x位置     y位置
...
第n个观测点的相机序号 第n个观察点的序号   第n个观测点的x位置 y位置
相机1
...
相机总数
点1
...
点的总数
索引号从0开始
每个相机被设置为9个参数，R,t,f,k1 and k2，旋转矩阵R使用了罗德里格公式，变成了一个向量
We use a pinhole camera model; the parameters we estimate for each camera area 
rotation R, a translation t, a focal length f and two radial distortion parameters k1 and k2. 
The formula for projecting a 3D point X into a camera R,t,f,k1,k2 is: 
P  =  R * X + t       (conversion from world to camera coordinates)
p  = -P / P.z         (perspective division)
p' =  f * r(p) * p    (conversion to pixel coordinates)
where P.z is the third (z) coordinate of P. In the last equation, 
r(p) is a function that computes a scaling factor to undo the radial distortion: 
r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.
This gives a projection in pixels, 
where the origin of the image is the center of the image, 
the positive x-axis points right, 
and the positive y-axis points up 
(in addition, in the camera coordinate system, 
the positive z-axis points backwards, 
so the camera is looking down the negative z-axis, as in OpenGL). 
'''


FILE_NAME = "problem-49-7776-pre.txt"
#FILE_NAME = 'problem-21-11315-pre.txt'
#Now read the data from the file:
def read_bal_data(file_name):
    with open(file_name, "rt") as file:
        #n_cameras 相机数目 49
        #n_points 点的数目 7776
        #n_observation 观察数目 31843
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())
        #str.split(str="", num=string.count(str))参数：
        #str - - 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
        #num - - 分割次数。默认为 - 1, 即分隔所有。

        # np.empty()返回一个随机元素的矩阵，大小按照参数定义
        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]
        camera_params = np.empty(n_cameras * 9)

        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))
        #print('camera_params2:{}'.format(camera_params))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))
    print('begin return shape:')
    print(camera_params.shape)
    print(points_3d.shape)
    print(camera_indices.shape)
    print(point_indices.shape)
    print(points_2d.shape)
    print('end return shape ')
    return camera_params, points_3d, camera_indices, point_indices, points_2d
"""
(相机数目，9)
(特征点的数目，3)
(观察到的点的数目，) 指在各个相机中观察到的点的数目之和，这些点有重合
(观察到的点的数目，)
(观察到的点的数目，2)
"""
camera_params = np.array([[1.574151594294026166e-02 ,-1.279093616385064240e-02 ,-4.400849808198078854e-03
                             ,-3.409383957718658403e-02 ,-1.075138710492152538e-01 ,1.120224029123603193e+00,
                          3.997515263935843564e+02 ,-3.177064385280357867e-07, 5.882049053459402224e-13],
                        [2.574151594294026166e-02 ,-1.279093616385064240e-02 ,-4.400849808198078854e-03
                             ,-4.409383957718658403e-02 ,-8.075138710492152538e-01 ,5.120224029123603193e+00,
                          1.997515263935843564e+02 ,-4.177064385280357867e-07, 5.882049053459402224e-13]])
points_3d = np.array([[-6.120001571722636369e-01, 5.717590477602828569e-01, -1.847081276454882293e+00],
                    [-2.120001571722636369e-01, 3.717590477602828569e-01, -0.847081276454882293e+00]])
camera_indices = np.array([1,0,0])
point_indices = np.array([0,1,1])
points_2d = np.array([[-3.326499999999999773e+02, 2.620899999999999750e+02],
                      [-4.326499999999999773e+02, 1.620899999999999750e+02],
                      [-0.326499999999999773e+02, 1.620869999999999750e+02]])

#camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

if True:
    print('begin return shape:')
    print(camera_params.shape)
    print(points_3d.shape)
    print(camera_indices.shape)
    print(point_indices.shape)
    print(points_2d.shape)
    print('end return shape ')
n_cameras = camera_params.shape[0]
n_points = points_3d.shape[0]
n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))

def rotate(points, rot_vecs):
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    #print('retate:{}'.format(cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v))
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
    return A

#Now we are ready to run optimization.
# Let's visualize residuals evaluated with the initial parameters.


def main():
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    t0 = time.time() #计时开始
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {} seconds".format(t1 - t0))

    # Setting scaling='jac' was done to automatically scale the variables and
    # equalize their influence on the cost function
    # (clearly the camera parameters and coordinates of the points are very different entities).
    # This option turned out to be crucial for successfull bundle adjustment.
    # Now let's plot residuals at the found solution:
    plt.plot(res.fun)

main()


'''
数据格式
相机数目    点的数目    观察数目
相机序号    点的序号    x位置     y位置
    ...
第n个观测点的相机序号 第n个观察点的序号   第n个观测点的x位置 y位置
相机1
相机数目
点1
点的数目
索引号从0开始
每个相机被设置为9个参数，R,t,f,k1 and k2，旋转矩阵R使用了罗德里格公式，变成了一个向量
(21, 9)
(11315, 3)
(36455,)
(36455,)
(36455, 2)

(49, 9)
(7776, 3)
(31843,)
(31843,)
(31843, 2)

(1, 9)
(1, 3)
(1,)
(1,)
(1, 2)

(相机数目，9)
(特征点的数目，3)
(观察点的数目，) 指在各个相机中观察到的点的数目之和，这些点有重合
(观察到的点的数目，)
(观察到的点的数目，2)
'''