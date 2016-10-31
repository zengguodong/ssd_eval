# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import os
import cPickle
import numpy as np
from conf.config import cfg
import caffe
from utils.Timer import Timer
from nms.cpu_nms import cpu_nms
from voc_eval import voc_eval

class ImdbEval(object):
    def __init__(self, imdb_name):
        # VOC_2007_test
        self.image_set, self.year ,self.name = imdb_name.split('_')
        # self.root_dir = None
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.root_dir = os.path.join(self.cur_dir, '..', '..')
        
        #where is the dataset
        self._init_imgset_classes()

        # where to put the result file
        self._init_res_path()
       

        self._init_general_config()
        
        self._init_caffe_config()
        


    def _init_imgset_classes(self):
        '''
        This function to init the parameters as below:
            # self.image_index = None
            # self.num_img = None
            # self.classes = None
            # self.num_classes = None
        '''
        pass

    def _init_res_path(self):
        '''
        This function to init the parameters as below:
            # the dir name to save the result file
            # self.res_dir_name = None
            # self.res_file_ext = None
            # self.all_boxes = None
        '''
        pass

    
    def _init_general_config(self):
        '''
        This function to init the parameters as below:
            self.image_resize = 300 
            self.conf_thresh = 0.01
            self.nms_thresh = 0.5  
            self.img_mean_pixel = [104,117,123]
            # self.model_def = None
            # self.model_weights = None
        '''

        

    def _return_img_mean_pixel(self):
        return None


    def _init_caffe_config(self):
        '''
        This function to init the parameters as below:
            
        '''
        if cfg.GUP_MODE:
            caffe.set_mode_gpu()
            caffe.set_device(cfg.GPU_ID)
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Net(self.model_def,      # defines the structure of the model
                        self.model_weights,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)
        self.net.blobs['data'].reshape(1,3,self.image_resize,self.image_resize)
        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))

        mean_pixel = self._return_img_mean_pixel()
        if not self.img_mean_pixel:
            self.img_mean_pixel = [104,117,123]
        self.transformer.set_mean('data', np.array(self.img_mean_pixel)) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)  
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2,1,0))  

        # save the caffe model detection result
        self.all_boxes = [ [ np.empty((0,5), float) for _ in xrange(self.num_img)] 
                                    for _ in xrange(self.num_classes)]


        

    def detection_all(self):
        pass


    def  write_results_file(self):

        # write to detections.pkl
         # self.res_txt_save_tmplate
            # self.res_pkl_save_file
        pkl_path = self.res_pkl_save_file
        with open(pkl_path, 'wb') as f:
            print 'Wrting pkl file: {}', self.res_pkl_save_file
            cPickle.dump(self.all_boxes, f, cPickle.HIGHEST_PROTOCOL)

        # write to individual result txt files
        for cls_ind, cls_name in enumerate(self.classes):
            print 'Writing {} {} results file'.format(cls_name, self.image_set)
            filename = self.res_txt_save_tmplate.format(cls_name)
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
        pass


    


    def _cal_all_boxes_len(self):
        _sum = 0
        for i in xrange(self.num_classes): 
            for j in xrange(self.num_img):
                _sum += len(self.all_boxes[i][j]) 
        return _sum

    
    def _read_res_files(self):
        # first try to read pickle file
        det_file = self.res_pkl_save_file
        if os.path.exists(det_file):
            print 'detecion file {:s} already exists, load directly '.format(det_file)
            with open(det_file, 'r') as f:
                self.all_boxes = cPickle.load(f)
                # print '*** detecion***, len=',self._cal_all_boxes_len()
                # print self.all_boxes[0][0]
            return True
        else:
            print 'detecion pkl file {:s} does not exists '.format(det_file)

        # there is no pickle file, read txt files
        for cls_idx, _class_name in enumerate(self.classes):
            cls_res_path = self.res_txt_save_tmplate.format(_class_name)
            if not os.path.exists(cls_res_path):
                raise Exception('File not exist:{:s}'.format(cls_res_path))
            else:
                print '>> read cls text file res: {}'.format(cls_res_path)

            with open(cls_res_path, 'r') as f:
                res_lines = f.readlines()
                res_lines = [line.strip().split(' ') for line in res_lines]

            for res_line in res_lines:
                file_name = res_line[0]
                conf = float(res_line[1])
                xmin, ymin = float(res_line[2]), float(res_line[3])
                xmax, ymax = float(res_line[4]),float(res_line[5])
                img_idx = self.image_index.index(file_name)
                if img_idx == -1:
                    raise Exception(" img name :{:s} does not exist in imaset file"\
                                                    .format(file_name))
                self.all_boxes[cls_idx][img_idx] = np.vstack(
                                        (self.all_boxes[cls_idx][img_idx],
                                            [xmin,ymin, xmax, ymax, conf]) )

        
        with open(det_file, 'wb') as f:
            cPickle.dump(self.all_boxes, f, cPickle.HIGHEST_PROTOCOL)

        return True

    def do_box_size_filter(self, max_box_size, min_box_size, read_res_file=False):
        if read_res_file:
            self._read_res_files()

        
        print '>> do box filter: min_size={}, max_size={}'.format(min_box_size, 
                                                                    max_box_size)
        print '>> before box size filter, sum of boxes = ', self._cal_all_boxes_len()
        for i in xrange(self.num_classes): 
            for j in xrange(self.num_img):
                if len(self.all_boxes[i][j]) > 0:
                    boxes = np.array(self.all_boxes[i][j])
                    _width = boxes[:,2] - boxes[:,0] 
                    _height = boxes[:,3]-boxes[:,1]
                    _width_filter =( (_width<=max_box_size) 
                                                & (_width>=min_box_size) )
                    _height_filter = ( (_height<=max_box_size)
                                                & (_height>=min_box_size) )
                    filter_conf = np.array((_width_filter & _height_filter), dtype='bool')
                    
                    self.all_boxes[i][j] = boxes[filter_conf]
        print '>> after box size filter, sum of boxes = ', self._cal_all_boxes_len()
        # xmin,ymin, xmax, ymax


    def do_conf_filter_thresh(self, read_res_file=False):
        if self.conf_thresh <  0 :
            raise Exception("In do confidence filter thresh, the pramater must larger than 0.0")

        if read_res_file:
            self._read_res_files()

        print '>> before confidence thresh, sum of boxes = ', self._cal_all_boxes_len()
        for i in xrange(self.num_classes): 
            for j in xrange(self.num_img):
                if len(self.all_boxes[i][j]) > 0:
                    boxes = np.array(self.all_boxes[i][j])
                    filter_conf = np.array(boxes[:,-1] > self.conf_thresh, dtype='bool')
                    self.all_boxes[i][j] = boxes[filter_conf]
        print '>> after confidence thresh, sum of boxes = ', self._cal_all_boxes_len()


    def do_nms(self, read_res_file=False):
        if self.nms_thresh <  0 :
            raise Exception("In do confidence filter thresh, the pramater must larger than 0.0")

        if read_res_file:
            self._read_res_files()

        print '>> before do nms, sum of boxes = ', self._cal_all_boxes_len()
        # self._all_boxes = py_cpu_nms(self._all_boxes, _iou_thresh)
        _time = Timer("do_nms")
        _time.tic()

        # num_classes = len(self.all_boxes)
        # num_images = len(self.all_boxes[0])
        nms_boxes = [[ np.empty((0,5), float) for _ in xrange(self.num_img)]
                     for _ in xrange(self.num_classes)]
        
        for cls_ind in xrange(self.num_classes):
            for im_ind in xrange(self.num_img):
                dets = np.array(self.all_boxes[cls_ind][im_ind])
                dets = dets.astype(np.float32)
                if dets.shape[0] == 0:
                    continue
                # CPU NMS is much faster than GPU NMS when the number of boxes
                # is relative small (e.g., < 10k)
                # TODO(rbg): autotune NMS dispatch
                # print 'dets.shape', dets.shape
                keep = cpu_nms(dets, self.nms_thresh)
                if len(keep) == 0:
                    continue
                nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
        # return nms_boxes
        self.all_boxes = nms_boxes
        _time.toc()
        print '>> after do nms, sum of boxes = ', self._cal_all_boxes_len()

    def do_eval(self, read_res_file=False):
        annopath = os.path.join(self.data_dir, 'Annotations', '{:s}.xml')
        imagesetfile = self.imgset_path
        print '** annothpath=', annopath
        print '** imagesetfile=', imagesetfile

        if read_res_file:
            self._read_res_files()
            
        if cfg.MAX_BOX_SIZE is None  or cfg.MIN_BOX_SIZE is None:
            raise Exception( '>>Please make sure cfg.MAX_BOX_SIZE and MIN_BOX_SIZE not None' )

        cachedir = os.path.join(self.data_dir, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self.year) < 2010 else False
        # print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(self.res_root_path):
            os.mkdir(self.res_root_path)

        res_analysis = {
            'num_box_scale':len(cfg.MAX_BOX_SIZE),
            'num_class':self.num_classes,
            'gt_box_num':[ [0 for _ in xrange(self.num_classes)] 
                                    for _ in xrange(len(cfg.MAX_BOX_SIZE))],
            'ap':[ [ 0.0 for _ in xrange(self.num_classes)] 
                                    for _ in xrange(len(cfg.MAX_BOX_SIZE))],
        }

        for i in xrange(len(cfg.MAX_BOX_SIZE)):
            max_box_size = cfg.MAX_BOX_SIZE[i]
            min_box_size = cfg.MIN_BOX_SIZE[i]
            self.do_box_size_filter(max_box_size*1.2, min_box_size)

            for j, cls in enumerate(self.classes):
                filename = self.res_txt_save_tmplate.format(cls)
                rec, prec, ap, gt_num = voc_eval(
                                    filename, annopath, imagesetfile, cls, cachedir, 
                                    ovthresh=self.nms_thresh,
                                    use_07_metric=use_07_metric, 
                                    max_box_size=max_box_size,
                                    min_box_size=min_box_size)

                res_analysis['ap'][i][j] = ap
                res_analysis['gt_box_num'][i][j] = gt_num
                # print'AP for {} = {:.4f}, gt_num={}'.format(cls, ap, gt_num) 
                print 'result for cls {} : len is {}, gt_num={}, rec[gt_num*5]={}, ap={}'.format (cls,
                                                        len(rec), gt_num, rec[-1], ap)

        with open(os.path.join(self.res_root_path, 'res_analysis.pkl'), 'w') as f:
            cPickle.dump(res_analysis, f)

        with open(os.path.join(self.res_root_path, 'res_analysis.txt'), 'w') as f:
            f.write('different box size: {}\n'.format(cfg.MAX_BOX_SIZE))

            for i, _class_name in enumerate(self.classes):
                f.write('{}: '.format(_class_name))

                for j in xrange(len(cfg.MAX_BOX_SIZE)):
                    ap = res_analysis['ap'][j][i] * 100
                    gt_num = res_analysis['gt_box_num'][j][i]
                    f.write('{:.1f}%({}) '.format(ap, gt_num))
                f.write('\n')

        print'AP for {} '.format( res_analysis['ap'][-1] ) 
        print'mean AP for {} '.format( np.mean(res_analysis['ap'][-1] ) )
        


        
       


class VocEval(ImdbEval):

    def __init__(self, imdb_name):
        super(VocEval, self).__init__(imdb_name)
        
    def _init_imgset_classes(self):
        '''
        This function to init the parameters as below:
            # self.image_index = None
            # self.num_img = None
            # self.classes = None
            # self.num_classes = None
        '''
        self.data_dir = os.path.join(self.root_dir, 'data', 'VOCdevkit', 
                                                    self.image_set+self.year)
        self.imgset_path = os.path.join(self.data_dir, 'ImageSets/Main', self.name+'.txt')
        with open(self.imgset_path, 'r') as f:
            lines = f.readlines()
            self.image_index = [line.strip() for line in lines]
            self.num_img = len(self.image_index)

        self.classes = cfg.VOC_CLASSES
        self.num_classes = len(self.classes)

    def _init_res_path(self):
        '''
        This function to init the parameters as below:
            # the dir name to save the result file
            # self.res_dir_name = None
            # self.res_file_ext = None
            # self.all_boxes = None

            # self.res_txt_save_tmplate
            # self.res_pkl_save_file  
        '''
        self.res_dir_name = cfg.VOC_RES_DIR_NAME
        self.res_txt_file_ext = cfg.VOC_RES_TXT_EXT
        self.res_pkl_file_name = cfg.VOC_RES_PKL_NAME
        if not self.res_dir_name or not self.res_txt_file_ext or not self.res_pkl_file_name:
            raise Exception('res_format , res_dir_name,res_file_name '
                    + 'could not be None when init class VOCResultProcess..')

        self.res_root_path = os.path.join(self.data_dir, 'results',
                        self.image_set+str(self.year),self.res_dir_name)

        if not os.path.exists(self.res_root_path):
            os.makedirs(self.res_root_path)

        self.res_txt_save_tmplate = os.path.join(self.res_root_path, 
                                                self.res_txt_file_ext + '_{:s}.txt')
        self.res_pkl_save_file = os.path.join(self.res_root_path, 
                                                self.res_pkl_file_name + '.pkl')


    def _return_img_mean_pixel(self):
        return cfg.VOC_IMG_MEAN_PIXEL

    def _init_general_config(self):
        '''
        This function to init the parameters as below:
            self.image_resize = 300 
            self.conf_thresh = 0.01
            self.nms_thresh = 0.5 

            self.img_mean_pixel = [104,117,123]
            # self.model_def = None
            # self.model_weights = None 
        '''
        self.image_resize = cfg.VOC_IMAGE_SIZE
        self.conf_thresh = cfg.VOC_CONF_THRESH
        self.nms_thresh = cfg.VOC_NMS_THRESH 

        self.img_mean_pixel = cfg.VOC_IMG_MEAN_PIXEL

        self.model_def = os.path.join(self.root_dir, cfg.MODEL_DIR, cfg.MODEL_DEF)
        self.model_weights = os.path.join(self.root_dir, cfg.WEIGHTS_DIR, cfg.WEIGHTS_DEF)
    

        # read classes

        

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

    def detection_all(self):
        try:
            self._read_res_files()
            print '>> detection, result file already exists... does not do detection.. '
            return
        except:
            print '>> no result file.. do detection for each image..'

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


def main():
    ssd = TestSSD('VOC_2007_test')
    ssd.detection_out()
    # ssd.detection_all()
    # img_path = '/home/guodong/dpln/ssd_caffe/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg'
    # print ssd.detect_one_img(img_path)

    # ssd.write_results_file()