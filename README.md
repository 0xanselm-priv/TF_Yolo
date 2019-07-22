# Yolo Object Detection for Presentation 07/16


### Pip Files
* tensorflow (tf)
* numpy (np)
* Pillow (pil)
* opencv-python (cv2)
* seaborn

### Download pretrained weights
```
https://pjreddie.com/media/files/yolov3.weights
```

### How to run model
Please make sure to load and store the weights in `./data` Afterwards put some images as `*.jpg`  in the `./data/images` folder
and run
```
python detect.py
```
The detections will be stored in `./detections`.

## Acknowledgments
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
* [A Tensorflow Slim implementation](https://github.com/mystic123/tensorflow-yolo-v3)
* [ResNet official implementation](https://github.com/tensorflow/models/tree/master/official/resnet)
* [Inspiration and Fundamentals](https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow)
