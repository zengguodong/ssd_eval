cd /home-2/wliu/projects/caffe
./build/tools/caffe test \
--solver="models/VGGNet/VOC0712/SSD_500x500/solver.prototxt" \
--weights="models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0,1,2,3 2>&1 | tee jobs/VGGNet/VOC0712/SSD_500x500/VGG_VOC0712_SSD_500x500.log
