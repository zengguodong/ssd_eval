import _init_paths
from conf.config import cfg
from eval_recall.imdb_eval import VocEval
import caffe

def main():
	# print cfg.VOC_IMG_MEAN_PIXEL
	# print type(cfg.VOC_IMG_MEAN_PIXEL)
	voc_eval = VocEval('VOC_2007_test')
	voc_eval.detection_all()
	voc_eval.do_conf_filter_thresh()
	voc_eval.do_nms()
	# voc_eval.write_results_file()
	voc_eval.do_eval()

if __name__=='__main__':
	main()