import numpy as np
import cv2
import math
from AM09_odometry import PinholeCamera, VisualOdometry
import random
import pandas as pd
# cam = PinholeCamera(1200,900,7.18856e+02 ,7.18856e+02 ,6.071928e+02 ,1.852157e+02)
cam = PinholeCamera(1200, 900, 3258.52325987820, 3221.05397172884, 626.110732333212, 439.400007275847)
# cam = PinholeCamera(1200,900,766.765413391747,763.363755865027,162.478636966162, 134.075559387391)
# annotations_path = 'D:\AM09/AM09posexy_zero.txt'
annotations_path = 'D:\AM02/AM02posexy_zero.txt'
# annotations_path = 'D:\kaist_place\PM10/PM10_East_posexy.txt'
vo = VisualOdometry(cam, annotations_path)

# vo = VisualOdometry(cam, 'D:\AM02/AM02posexy_zero.txt')
# vo = VisualOdometry(cam, 'D:\dataset_all\KITTI_dataset_22G\dataset\sequences\poses/00.txt')
degree_xita = 120
degree_xita_calculate = 5
xita = 3.1415/180*(degree_xita)  # 55degree  120
xita_calculate = 3.1415/180*(degree_xita_calculate)  # 25degree
traj = np.zeros((1000,1000,3), dtype=np.uint8)
DEFAULT = 0
s =0.0
matches = 0
inliers = 0
window_control = 1
stride = 100
# begin = 6300 stop = 8000 AM09seq
begin = 0
stop = 8000
with open(annotations_path) as f:
    annotations = f.readlines()
    begin_location = annotations[begin].strip().split()
    x_prev = (float(begin_location[0]))
    z_prev = (float(begin_location[1]))
# print('begin_location:{}'.format(x_prev))
# begin = 6250
# stop = 6450
for img_id in range(8859):
    id = img_id*stride+begin
    # if id==1575or id==1656or id==1891or id==1893or(id>1689and id <1720)or id==1957or id==1995or id==2043\
    # 		or id==2067or id==2069or id==2078or id==2095:
    # 	continue
    # print(id)
    if id > stop:
        break
    # img = cv2.imread('D:\DWT_test\AM09_fused_DWT/'+str(id)+'.png',0)
    # img = cv2.imread('D:\AM09\AM09_vi/I'+str(id).zfill(5)+'.png',0)
    # img = cv2.imread('./AM02Dense_epoch10/'+str(id)+'.png', 0)
    # img = cv2.imread('D:\dataset_all\KITTI_dataset_22G\dataset\sequences/00\image_0/'+str(id).zfill(6)+'.png',0)
    img = cv2.imread('D:\AM09\AM09_vi/I'+str(id).zfill(5)+'.png',0)
    # img = cv2.imread('D:/kaist_place/PM10/East/East_image/vi/I'+str(id).zfill(5)+'.png',0)
    # img = cv2.imread('D:\AM09\AM09_epoch29/F9_' + str(id).zfill(2) + '.png', 0)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    vo.update(img, id,DEFAULT, stride)
    DEFAULT += 1
    cur_t = vo.cur_t
    Fast_threshold = vo.Fast_threshold
    if img_id > 2:
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
    else:
        x, y, z = 0, 0, 0

    # vo.trueX = -vo.trueX
    a = -x_prev
    b = -z_prev
    # a = 0
    # b = 0
    # print(math.cos(xita))
    trueX = (math.cos(xita)*(vo.trueX+a)-math.sin(xita)*(vo.trueZ+b))
    trueZ = math.sin(xita)*(vo.trueX+a)+math.cos(xita)*(vo.trueZ+b)
    # trueX = vo.trueX
    # trueZ = vo.trueZ
    draw_x = -(math.cos(xita_calculate) * (x) - math.sin(xita_calculate) * (z))
    draw_y = math.sin(xita_calculate) * (x) + math.cos(xita_calculate) * (z)
    # draw_x= x
    # draw_y= z

    true_x, true_y = int(trueX), int(trueZ)
    draw_x, draw_y = int(draw_x), int(draw_y)
    
    # print('truex,truey:{},{}\ndrawx,drawy{},{}'.format(true_x,true_y,draw_x,draw_y))
    picture_a = 100
    picture_b = 100
    
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "x=%0.2fm y=%0.2fm z=%0.2fm frame=%4d"%(x,y,z,id)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    text_img = "Coordinates: frame=%4d"%(id)
    cv2.putText(img, text_img, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
    if img_id > 3:
        cv2.circle(traj, (draw_x + picture_a, draw_y + picture_b), 2,(img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 2)
        cv2.circle(traj, (true_x + picture_a, true_y + picture_b), 2, (0, 0, 255), 2)
        matches += len(vo.key_points)
        inliers += vo.key_points_in
        for i in vo.key_points:
            cv2.circle(img, tuple(i), 1, (255, 255, 255), 2)
    if window_control:
        cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', traj)
    s = vo.S
    cv2.waitKey(1)
cv2.imwrite('AM_02_ir_result'+str(random.random())+'.png', traj)

np_data_t = vo.t_plot[:,0:2]
np_data_R = vo.R_plot[:,0:2]
np_data_t = vo.t_plot
np_data_R = vo.R_plot
pd_data_t = pd.DataFrame(np_data_t,columns=['t1','t2','t3'])
pd_data_R = pd.DataFrame(np_data_R,columns=['R1','R2','R3'])
pd_data_t.to_csv('./t_pd_data'+str(random.random())+'.csv')
pd_data_R.to_csv('./R_pd_data'+str(random.random())+'.csv')
print('draw:{},{}'.format(float(draw_x), float(draw_y)))
print('true:{},{}'.format(float(true_x), float(true_y)))
print('mistake:{}'.format(((float(draw_x)-float(true_x))**2+(float(draw_y)-float(true_y))**2)**0.5))
print('total distance:{}'.format(s))
print('matches:{}\ninliers:{}\npercentage:{}'.format(matches/9500,inliers/9500,inliers/matches))
print('degree_xita:{},degree_xita_calculate:{}'.format(degree_xita,degree_xita_calculate))
print('Fast_threshold :{}'.format(Fast_threshold ))
