import numpy as np

# Make sure that caffe is on the python path:
caffe_root = './'  
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_mode_cpu()
# caffe.set_device(0)

import cPickle
import os
from google.protobuf import text_format
from caffe.proto import caffe_pb2





# # load PASCAL VOC labels
# labelmap_file = '/home/guodong/dpln/ssd_caffe/data/VOC0712/labelmap_voc.prototxt'
# file = open(labelmap_file, 'r')
# labelmap = caffe_pb2.LabelMap()
# text_format.Merge(str(file.read()), labelmap)

# def get_labelname(labelmap, labels):
#     num_labels = len(labelmap.item)
#     labelnames = []
#     if type(labels) is not list:
#         labels = [labels]
#     for label in labels:
#         found = False
#         for i in xrange(0, num_labels):
#             if label == labelmap.item[i].label:
#                 found = True
#                 labelnames.append(labelmap.item[i].display_name)
#                 break
#         assert found == True
# #     return labelnames

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
					"cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
			 		"person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

class TestSSD(object):
	def __init__(self, imdb_name, img_size=300, conf_thresh=0.01, nms_thresh=0.5, 
						res_dir_name='', res_file_ext='VOC_det'):
		# VOC_2007_test
		self.image_set, self.year ,self.name = imdb_name.split('_')
		self.root_dir = os.path.dirname(os.path.realpath(__file__))
		
		self.image_index = None
		self.num_img = None
		self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
					"cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
			 		"person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
		self.num_classes = len(self.classes)
		self.all_boxes = None
		# the dir name to save the result file
		self.res_dir_name = res_dir_name
		self.res_file_ext = res_file_ext
		

		# set net to batch size of 1
		self.image_resize = img_size 
		self.conf_thresh = conf_thresh
		self.nms_thresh = nms_thresh
		self.size_extn = '{}x{}'.format(self.image_resize, self.image_resize)

		self.data_dir = os.path.join(self.root_dir, 'data', 'VOCdevkit', 
													self.image_set+self.year)
		self.model_def = os.path.join(self.root_dir, 'models/VGGNet/VOC0712', 
										'SSD_{}/deploy_conv4.prototxt'.format(self.size_extn))
		_mode_wights = 'VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'
		self.model_weights = os.path.join(self.root_dir, 'models/VGGNet/VOC0712',
											'SSD_{}'.format(self.size_extn), _mode_wights)

		self.init()

	def _init_label(self):
		self.labelmap_file = os.path.join(self.root_dir, 'data/VOC0712/labelmap_voc.prototxt')
		with open(self.labelmap_file, 'r') as f:
			self.labelmap = caffe_pb2.LabelMap()
			text_format.Merge(str(f.read()), self.labelmap)

	def get_labelname(self, labels):
		num_labels = len(self.labelmap.item)
		labelnames = []
		if type(labels) is not list:
		    labels = [labels]
		for label in labels:
		    found = False
		    for i in xrange(0, num_labels):
		        if label == self.labelmap.item[i].label:
		            found = True
		            labelnames.append(self.labelmap.item[i].display_name)
		            break
		    assert found == True
		return labelnames

	def _init_res_path(self):
		if not self.res_dir_name or not self.res_file_ext:
			raise Exception('res_format , res_dir_name,res_file_name '
			        + 'could not be None when init class VOCResultProcess..')

		self.res_root_path = os.path.join(self.data_dir, 'results',
		                self.image_set+str(self.year),self.res_dir_name)

		if not os.path.exists(self.res_root_path):
			os.makedirs(self.res_root_path)


	def _get_imgset_path(self, name='test'):
		return os.path.join(self.data_dir, 'ImageSets/Main', self.name+'.txt')

	def _read_imgset(self):
		imgset_path = self._get_imgset_path()
		with open(imgset_path, 'r') as f:
			lines = f.readlines()
			self.image_index = [line.strip() for line in lines]
			self.num_img = len(self.image_index)

	def _init_all_boxes(self):
		self._init_label()
		self._read_imgset()
		self._init_res_path()
		self.all_boxes = [ [ np.empty((0,5), float) for _ in xrange(self.num_img)] 
		                            for _ in xrange(self.num_classes)]

		

	def init(self):
		self.net = caffe.Net(self.model_def,      # defines the structure of the model
		                self.model_weights,  # contains the trained weights
		                caffe.TEST)     # use test mode (e.g., don't perform dropout)
		self.net.blobs['data'].reshape(1,3,self.image_resize,self.image_resize)
		# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
		self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		self.transformer.set_transpose('data', (2, 0, 1))
		self.transformer.set_mean('data', np.array([104,117,123])) # mean pixel
		# the reference model operates on images in [0,255] range instead of [0,1]
		self.transformer.set_raw_scale('data', 255)  
		# the reference model has channels in BGR order instead of RGB
		self.transformer.set_channel_swap('data', (2,1,0))  


		self._init_all_boxes()

	def detection_all(self):

		for img_idx, img in enumerate(self.image_index):
			img_path = os.path.join(self.data_dir, 'JPEGImages', img+'.jpg')
			print '>> detect {:d}/{:d} {:s}'.format(img_idx+1, self.num_img, img_path)
			boxes = self.detect_one_img(img_path)

			for box in boxes:
				class_name = box[-1]
				cls_idx = self.classes.index(class_name)
				if cls_idx == -1:
					raise Exception(" class name: {} does not exist".format(class_name))
			
				self.all_boxes[cls_idx][img_idx] = np.vstack( 
												(self.all_boxes[cls_idx][img_idx],
                                            		[box[0],box[1], box[2], box[3], box[4]]) )
		self.write_results_file()

	def _get_results_file_template(self, _ext='.txt'):
		if _ext == '.txt':
			filename = self.res_file_ext + '_{:s}.txt'
		else:
			filename = '{:s}.pkl'

		path = os.path.join(self.res_root_path,  filename)
		return path

	def  write_results_file(self):

		# write to detections.pkl
		pkl_path = self._get_results_file_template('.pkl').format('detections')
		with open(pkl_path, 'wb') as f:
			cPickle.dump(self.all_boxes, f, cPickle.HIGHEST_PROTOCOL)

		# write to individual result txt files
		for cls_ind, cls_name in enumerate(self.classes):
			print 'Writing {} VOC results file'.format(cls_name)
			filename = self._get_results_file_template().format(cls_name)
			with open(filename, 'wt') as f:
				for im_ind, img_name in enumerate(self.image_index):
					dets = np.array(self.all_boxes[cls_ind][im_ind])
					if dets.shape[0]==0:
						continue
					# the VOCdevkit expects 1-based indices
					for k in xrange(dets.shape[0]):
						f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
							format(img_name, dets[k,-1],
								dets[k,0]+1, dets[k,1] + 1, dets[k,2] + 1, dets[k,3] + 1))




	def detect_one_img(self, img_path):
		# img_path = '/home/guodong/dpln/ssd_caffe/examples/images/fish-bike.jpg'
		if not os.path.exists(img_path):
			raise Exception('{} does not exist'.format(img_path))

		image = caffe.io.load_image(img_path)
		transformed_image = self.transformer.preprocess('data', image)
		self.net.blobs['data'].data[...] = transformed_image
		detections = self.net.forward()['detection_out']

		# Parse the outputs.
		det_label = detections[0,0,:,1]
		det_conf = detections[0,0,:,2]
		det_xmin = detections[0,0,:,3]
		det_ymin = detections[0,0,:,4]
		det_xmax = detections[0,0,:,5]
		det_ymax = detections[0,0,:,6]

		# Get detections with confidence higher than 0.6.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.conf_thresh]

		top_conf = det_conf[top_indices]
		top_label_indices = det_label[top_indices].tolist()
		# top_labels = self.get_labelname(top_label_indices)
		# top_labels = get_labelname(labelmap, top_label_indices)
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]

		boxes = []
		for i in xrange(top_conf.shape[0]):
		    xmin = int(round(top_xmin[i] * image.shape[1]))
		    ymin = int(round(top_ymin[i] * image.shape[0]))
		    xmax = int(round(top_xmax[i] * image.shape[1]))
		    ymax = int(round(top_ymax[i] * image.shape[0]))
		    score = top_conf[i]
		    label = int(top_label_indices[i]-1)
		    label_name = self.classes[label]
		    boxes.append([xmin,ymin, xmax, ymax, score, label_name])

		# print len(boxes), ' boxes[:2] = ',  boxes[:2]
		return boxes


	def do_nms(self):
		pass

	def do_conf_filter(self):
		pass


def main():
	ssd = TestSSD('VOC_2007_test',conf_thresh=0.01, res_dir_name='ssd_300*300', res_file_ext='det_VOC')
	# ssd.detection_all()
	img_path = '/home/guodong/dpln/ssd_caffe/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
	# print ssd.detect_one_img(img_path)

	# ssd.write_results_file()

# load PASCAL VOC labels

def test():
	model_def = '/home/guodong/dpln/ssd_caffe/models/VGGNet/VOC0712/SSD_300x300/deploy_pool6.prototxt'
	# model_def = '/home/guodong/dpln/ssd_caffe/models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
	model_weights = '/home/guodong/dpln/ssd_caffe/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

	net = caffe.Net(model_def,      # defines the structure of the model
	                model_weights,  # contains the trained weights
	                caffe.TEST)     # use test mode (e.g., don't perform dropout)

	# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_mean('data', np.array([104,117,123])) # mean pixel
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
	
	# set net to batch size of 1
	image_resize = 300
	net.blobs['data'].reshape(1,3,image_resize,image_resize)
	# net.blobs['data'].reshape(1,3,image_resize,image_resize)

	
	for img in ['000006']:
		image_path = '/home/guodong/dpln/ssd_caffe/data/VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(img)
		image = caffe.io.load_image(image_path)
		transformed_image = transformer.preprocess('data', image)
		net.blobs['data'].data[...] = transformed_image
		# net.blobs['data'].data[...] = image

		# Forward pass.
		detections = net.forward()['detection_out']

		# Parse the outputs.
		det_label = detections[0,0,:,1]
		det_conf = detections[0,0,:,2]
		det_xmin = detections[0,0,:,3]
		det_ymin = detections[0,0,:,4]
		det_xmax = detections[0,0,:,5]
		det_ymax = detections[0,0,:,6]

		# Get detections with confidence higher than 0.6.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]

		top_conf = det_conf[top_indices]
		top_label_indices = det_label[top_indices].tolist()
		# print 'top_label_indices=',top_label_indices
		top_labels = get_labelname(labelmap, top_label_indices)
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]

		print 'detections', detections.shape
		for i in xrange(top_conf.shape[0]):
			# print 'i=',i
			xmin = int(round(top_xmin[i] * image.shape[1]))
			ymin = int(round(top_ymin[i] * image.shape[0]))
			xmax = int(round(top_xmax[i] * image.shape[1]))
			ymax = int(round(top_ymax[i] * image.shape[0]))
			score = top_conf[i]
			label = int(top_label_indices[i]-1)
			label_name = classes[label]
			display_txt = '%s: %.2f'%(label_name, score)
			coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
			# color = colors[label]
			# currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
			# currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
			print display_txt, coords

# chair: 0.90 ((127, 210), 248, 164)
# diningtable: 0.60 ((132, 201), 229, 175)
# sofa: 0.69 ((136, 198), 226, 168)

if __name__=='__main__':
	test()