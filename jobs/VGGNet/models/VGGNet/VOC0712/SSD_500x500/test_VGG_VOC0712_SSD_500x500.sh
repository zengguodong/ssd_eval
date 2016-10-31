cd /home-2/wliu/projects/caffe
./build/tools/caffe test \
--model="models/VGGNet/VOC0712/SSD_500x500/test.prototxt" \
--weights="models/VGGNet/VOC0712/VGG_VOC0712_SSD_500x500_iter_60000.caffemodel" \
--gpu 0,1,2,3 2>&1 | tee jobs/VGGNet/VOC0712/SSD_500x500/VGG_VOC0712_SSD_500x500.log
