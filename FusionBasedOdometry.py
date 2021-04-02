import numpy as np
import cv2
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21),
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2


class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy,
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]


class FusionBasedOdometry:
	def __init__(self, cam, annotations):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.n = 0
		self.S =0
		self.scale = 0
		self.key_points = None
		self.key_points_in = None
		self.lines = []
		self.cur_R = None
		self.Fast_threshold = 70
		self.pp = (cam.cx, cam.cy)
		self.t_plot = np.array([[0,0,0]])
		self.R_plot = np.array([[0,0,0]])
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		self.detector = cv2.FastFeatureDetector_create(threshold=self.Fast_threshold, nonmaxSuppression=True)
		with open(annotations) as f:
			self.annotations = f.readlines()

	def getAbsoluteScale(self, frame_id, n):  #specialized for KITTI odometry dataset

		ss = self.annotations[frame_id-n].strip().split()
		x_prev = (float(ss[0]))
		z_prev = (float(ss[1]))
		ss1 = self.annotations[frame_id].strip().split()
		x = (float(ss1[0]))
		z = (float(ss1[1]))
		self.trueX,  self.trueZ = x, z
		# print('truex truey:{},{}'.format(self.trueX,self.trueZ))
		# print(np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev)))
		self.S += (np.sqrt((x - x_prev)**2 + (z - z_prev)**2))
		# print('scale{}'.format(np.sqrt((x - x_prev)**2 + (z - z_prev)**2)))

		scale = np.sqrt((x - x_prev)**2 + (z - z_prev)**2)
		# if scale ==0:
		# 	scale = 0.15
		# 	print('frame:{}'.format(frame_id, ))
		return scale
		
	def processFirstFrame(self):
		self.px_ref = self.detector.detect(self.new_frame)
		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME
		self.px_ref = self.px_cur

	def processFrame(self, frame_id, stride):
		
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=2.0)
		self.key_points_in = len(self.px_cur[np.nonzero(mask)[0]])
		self.px_cur = self.px_cur[np.nonzero(mask)[0]]
		self.px_ref = self.px_ref[np.nonzero(mask)[0]]
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		absolute_scale = self.getAbsoluteScale(frame_id, stride)
		R_change, jac = cv2.Rodrigues(R)
		self.key_points = self.px_cur
		# self.cur_R = np.abs(self.cur_R)
		# t = np.abs(t)
		cur_R_t = self.cur_R.dot(t)
		# cur_R_t=np.fabs(cur_R_t)
		# print('t_plot:{},t:{}'.format(self.t_plot,t.reshape(1,3)))
		self.t_plot = np.concatenate((self.t_plot,t.reshape(1,3)),0)
		self.R_plot = np.concatenate((self.R_plot,R_change.reshape(1,3)),0)
		self.cur_t = self.cur_t + absolute_scale*cur_R_t
		kkkk = 1
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, frame_id, DEFAULT1, stride):
		# assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if frame_id > 7801:
			self.detector = cv2.FastFeatureDetector_create(threshold=40, nonmaxSuppression=True)
		if DEFAULT1==0:
			self.processFirstFrame()
		if DEFAULT1==1:
			self.processSecondFrame()
		if DEFAULT1 > 1:
			self.processFrame(frame_id,stride)
		self.last_frame = self.new_frame

