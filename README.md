# Evaluate for SSD on different layers and scales: Single Shot MultiBox Detector


### Contents
1. [Installation](#installation)
2. [Eval](#Eval)

### Installation
1. Get the code which mainly evaluates the precision and recall of SSD.
You can config the parameters to test on the result on different layers and object size.

  We will call the directory that you cloned ssd_eval into $SSD_EVAL_ROOT

 ```Shell
  git clone https://github.com/zengguodong/ssd_eval.git
  ```


2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make all
  # Make sure to include $SSD_EVAL_ROOT/python to your PYTHONPATH.
  make pycaffe

  # Compile blob nms
  cd $SSD_EVAL_ROOT/lib
  make all
  ```


### Eval
1.  Get VOC 2007 test data
  ```Shell
 cd $SSD_EVAL_ROOT/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtest_06-Nov-2007.tar
  ```

2. Download the model 
  ```Shell
cd  $SSD_EVAL_ROOT
wget http://www.cs.unc.edu/~wliu/projects/SSD/models_VGGNet_VOC0712_SSD_300x300.tar.gz
tar -xvf models_VGGNet_VOC0712_SSD_300x300.tar
  ```

3. Evaluate SSD on VOC2007 test data
  ```Shell
cd $SSD_EVAL_ROOT
python main.py
  ```

4. Change the parameter to test on different layers of SSD and object size

**test the box proposals only on different layers**

 change the paramter of __C.MODEL_DEF int the file of lib/conf/config.py, the options could be [deploy_conv4.prototxt, deploy_conv6.prototxt, deploy_conv7.prototxt,deploy_conv8.prototxt, deploy_fc7.prototxt or deploy.prototxt]

**test SSD for different size of object**

change the parameter of __C.MAX_BOX_SIZE and __C.MIN_BOX_SIZE. 