import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import plot,savefig
# img1 = cv2.imread('D:/AM09/East_image/vi/I00255.png',0)
# cv2.imwrite('D:/img1.png',img1)
# #
# # detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
# # pre_frame = cv2.imread('D:\\opencv_test\\00.png', 0)
# # cur_frame = cv2.imread('D:\\opencv_test\\01.png', 0)
# # lk_params = dict(winSize=(21, 21),  # 用于光流法的参数
# #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# #
# #
# # def feature_tracking(image_ref, image_cur, px_ref):#特征追踪
# #     kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
# #     st = st.reshape(st.shape[0])
# #     kp1 = px_ref[st == 1]
# #     kp2 = kp2[st == 1]
# #     # print('kp1 shape:{}'.format(kp1.shape))
# #     # print('kp2.shape:{}'.format(kp2.shape))
# #
# #     return kp1, kp2,err
# #
# #
# # K = np.array([[718.856, 0., 607.1928], [0., 718.856, 185.2157], [0., 0., 1.]])
# # kp_pre = detector.detect(pre_frame)
# # kp_pre = np.array([x.pt for x in kp_pre], dtype=np.float32)
# # kp_pre, kp_cur, _diff = feature_tracking(pre_frame, cur_frame, kp_pre)
# # # print(kp_cur)
# # E, mask = cv2.findEssentialMat(kp_cur, kp_cur,
# #                                K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
# # # kp_cur = kp_cur[np.nonzero(mask)[0]]
# # _, R, t, mask = cv2.recoverPose(E, kp_cur, kp_cur, K)
# # # print(R)
# # # print(t)
# # # print(kp_cur)
# # # sum = 0
# # # k = 0
# # # err = 0
# # # for i in mask:
# # #     # print(i)
# # #     k += 1
# # #     if i == np.array([1]):
# # #         sum += 1
# # #     if i == np.array([0]):
# # #         err += 1
# # # print(sum)
# # # print(k)
# # # print(err)
# # # print(mask.shape)
# # # print(kp_cur.shape)
# # # b = np.nonzero(mask)[0]
# # # for i in b:
# # #     print(pre_frame[i])
# # # [[-2.25792560e-03 -7.06845043e-01 -1.69878053e-02]
# # #  [ 7.06827600e-01 -2.28585692e-03 -8.09657047e-03]
# # #  [ 1.80270139e-02  8.68284997e-03  3.90027298e-06]]
# # # R = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype = float)
# # # R = np.array([0,0,0],dtype = float)
# # R = np.array([0.1, 0.2, 0.5], dtype=float)
# # # R0 = np.array([[0.70267181, 0.67913767, -0.21218943],
# # #             [-0.63812689, 0.7334299, 0.23425342],
# # #             [0.31471639, -0.02919949, 0.94873652]])
# # # R_change,jac = cv2.Rodrigues(R0)
# # # print(R_change)
# # # R2 = np.array([[-0.14714744], [-0.29429488], [-0.73573721]])
# # # R1, _ = cv2.Rodrigues(R2)
# # # R3, _ = cv2.Rodrigues(R1)
# # # print(R1)
# # # print(R3)
# # R,jac = cv2.Rodrigues(R)
# # R,jac = cv2.Rodrigues(R)
# # print(R)

# 取代符号
# old_str = "," #老文件内容字段
# new_str = " " #要改成字段
# file_data = ''
# with open('posexyz.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         if old_str in line:
#             line = line.replace(old_str, new_str)
#             file_data += line
# with open('poseall.txt', 'w',encoding='utf-8') as f:
#     f.write(file_data)

# 画轨迹
# import cv2
# import numpy as np
# traj = np.zeros((1080,1920,3), dtype=np.uint8)  # 构建一幅空白图片
# with open('D:\AM09_posexyz.txt', 'r', encoding='utf-8') as pose:
#     # for line in pose:
#     #     x = eval(line.strip().split()[0])
#     #     y = eval(line.strip().split()[1])
#     #     z = eval(line.strip().split()[2])
#     for frame_id in range(1000):
#         cv2.circle(traj, (int(x)+800,int(y)+100), 1, (0,0,255), 1)
#         cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
#         text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)  # 真实值
#         cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
#         cv2.imshow('Trajectory', traj)
#         cv2.waitKey(100)
# cv2.imwrite('map.png', traj)
from PIL import Image

# ir_img1 = cv2.imread('./image_test/ir375.png', 0)
# vi_img1 = cv2.imread('./image_test/vi375.png', 0)
# ir_img2 = cv2.imread('./image_test/ir376.png', 0)
# vi_img2 = cv2.imread('./image_test/vi376.png', 0)
_img_id = 8103
ir_img1 = cv2.imread('D:\AM09/'+str(_img_id)+'ir.png', 0)
ir_img2 = cv2.imread('D:\AM09/'+str(_img_id+1)+'ir.png', 0)
vi_img1 = cv2.imread('D:\AM09/'+str(_img_id)+'vi.png', 0)
vi_img2 = cv2.imread('D:\AM09/'+str(_img_id+1)+'vi.png', 0)
result_img1 = cv2.imread('D:\AM09/'+str(_img_id)+'fused.png', 0)
result_img2 = cv2.imread('D:\AM09/'+str(_img_id+1)+'fused.png', 0)
CLAHE = cv2.createCLAHE(clipLimit=3.0)
# result_img1 = CLAHE.apply(result_img1)
# result_img2 = CLAHE.apply(result_img2)
color = np.random.randint(0, 255, (50000, 3))
lk_params = dict(winSize=(21, 21),  # 用于光流法的参数
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# mask = result_img1
# cv2.imwrite('./image_test/mask2.png',mask)
cv2.imwrite('./'+str(_img_id+1)+'_fused_CLAHE.png',result_img2)
if 1:
	# old_gray = result_img1
	# fast = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)
	# p0 = fast.detect(old_gray ,None)
	# p0 = np.array([x.pt for x in p0], dtype=np.float32)
	# frame_gray = result_img2
	# # calculate optical flow
	# p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	# print(st.shape)
	# K = np.array([[3258.52325987820, 0, 626.110732333212], [0, 3221.05397172884, 439.400007275847], [0, 0, 1]])
	# # Select good points
	# print(p1.shape)
	# good_new = p1[np.nonzero(st)[0]]
	# good_old = p0[np.nonzero(st)[0]]
	# E, mask1 = cv2.findEssentialMat(good_new, good_old,
	#                                K, method=cv2.RANSAC, prob=0.99, threshold=0.5)
	# good_new = p1[np.nonzero(mask1)[0]]
	# good_old = p0[np.nonzero(mask1)[0]]
	# good_new = p1[st==[1]]
	# good_old = p0[st==[1]]
	# print(p1.shape)
	# print(good_new.shape)
	# draw the tracks
	# im = Image.open('./image_test/F9_01.bmp')
	# res_im2 =im
	im = Image.open('./8104_fused_CLAHE.png')
	# im = CLAHE.apply(im)
	# im = Image.open('./image_test/vi376_gray.png')
	rgb = im.convert('RGB')  # 灰度转RGB
	old_gray = result_img1
	fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
	p0 = fast.detect(old_gray, None)
	p0 = np.array([x.pt for x in p0], dtype=np.float32)
	frame_gray = result_img2
	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	print(st.shape)
	K = np.array([[3258.52325987820, 0, 626.110732333212], [0, 3221.05397172884, 439.400007275847], [0, 0, 1]])
	# Select good points
	print(p1.shape)
	# good_new = p1[np.nonzero(st)[0]]
	# good_old = p0[np.nonzero(st)[0]]
	good_new = p1
	good_old = p0
	E, mask1 = cv2.findEssentialMat(good_new, good_old,
	                                K, method=cv2.RANSAC, prob=0.995, threshold=0.5)
	# good_new = p1[np.nonzero(mask1)[0]]
	# good_old = p0[np.nonzero(mask1)[0]]
	# rgb.save('./image_test/frame01.png')
	rgb = np.float32(rgb)
	print(p1.shape)
	print(good_new.shape)
	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = new.ravel()
		c, d = old.ravel()
		# color[i].tolist()
		mask = cv2.line(rgb, (a, b), (c, d), color[i].tolist(), 3)
	# mask = cv2.line(img_CLAHE1, (a,b),(c,d), (155,155,155), 1)
	# frame = cv2.circle(frame_gray,(a,b),5,(i,255,255),-1)
	# img = np.concatenate([mask,frame],axis=1)
	# cv2.imshow('fig',mask)
	
	# cv2.waitKey(1000)
	
	cv2.imwrite('./image_test/' + str(random.random()) + '.png', mask)
# (5035, 2)
# (351, 2) without CLAHE
# (311, 2)

# 画出Fast特征点
# img = img_CLAHE
# fast = cv2.FastFeatureDetector_create(threshold=35, nonmaxSuppression=True)
# kp = fast.detect(img,None)
# img2 = cv2.drawKeypoints(img, kp, None,color=(0,0,255))
# cv2.imwrite('F9_01_CLAHE_feature.png',img2)
# end 画出Fast特征点

# CLAHE = cv2.createCLAHE(clipLimit=3.0)
# img_CLAHE = CLAHE.apply(img_CLAHE)
# cv2.imwrite('./image_test/vi376_CLAHE.png',img_CLAHE)
#
# detector = cv2.FastFeatureDetector_create(threshold=0, nonmaxSuppression=True)
# # K = np.array([[3258.52325987820,0,626.110732333212],[0,3221.05397172884,439.400007275847],[0,0,1]])
# lk_params = dict(winSize=(21, 21),  # 用于光流法的参数
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# def featureTracking(image_ref, image_cur, px_ref):#特征追踪
#     kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)
#     st = st.reshape(st.shape[0])
#     kp1 = px_ref[st == 1]
#     kp2 = kp2[st == 1]
#     return kp1, kp2, err  # 返回特征点位置
#
# px_ref = detector.detect(vi_img1)
# px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
# px_ref, px_cur, px_diff = featureTracking(vi_img1, vi_img2, px_ref)
# for i in px_ref:
#     cv2.circle(vi_img2, tuple(i), 1, (255, 255, 255), 2)
# p1, st, err = cv2.calcOpticalFlowPyrLK(vi_img1, vi_img2, p0, None, **lk_params)
# Select good points
# good_new = p1[st == 1]
# good_old = p0[st == 1]
# draw the tracks
# for i,(new,old) in enumerate(zip(px_ref,px_cur)):
#     a,b = new.ravel()
#     c,d = old.ravel()
#     vi_img1 = cv2.circle(vi_img1,(a,b),5,(0,0,255),-1)
# plt.imshow(vi_img1),plt.show()
# cv2.imwrite('./image_test/vi376_feature_track.png',vi_img2)

# 特征匹配 ORB
# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# from PIL import Image
# img1 = vi_img1
# img2 = vi_img2
#
# CLAHE = cv2.createCLAHE(clipLimit=3.0)
# img1 = CLAHE.apply(img1)
# img2 = CLAHE.apply(img2)
# # Initiate ORB detector
# orb = cv.ORB_create()
# fused_img1 = cv2.imread('./image_test/F9_00.png',0)
# fused_img2 = cv2.imread('./image_test/F9_01.png',0)
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# #BRIEF描述子
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1,des2)
# #按距离顺序存储
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 50 matches.画出开始的五十个
# print(kp1)
# print('\n')
# print(matches)
# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()
# cv2.imwrite('./image_test/fused_feature_track.png',img3)
#
# a = np.concatenate([vi_img2,img2],axis = 1)
#
# cv2.imwrite('a.png',a)

