cd /home/guodong/dpln/ssd_caffe
./build/tools/caffe test \
--model="models/VGGNet/VOC0712/SSD_300x300_score/deploy.prototxt" \
--weights="models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/VOC0712/SSD_300x300_score/VGG_VOC0712_SSD_300x300_test60000.log
