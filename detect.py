"""Yolo image detector.
Copy images to run Yolo Detector on in data/image folder
"""

import tensorflow as tf
import sys
import cv2
import os

from yolo_darknet import Yolo
import utility_functions
import load_weights

MODEL_SIZE = (416, 416)
CLASS_NAMES_FILE = './data/coco.names'
WEIGHT_FILE = './data/yolov3.weights'
IMAGES_DIR = './data/images/'
MODEL_FILE = './data/model.ckpt.index'
MAX_OUTPUT_SIZE = 20
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(type, iou_threshold, confidence_threshold, input_names):
    class_names = utility_functions.load_class_names(CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo(n_classes=n_classes, model_size=MODEL_SIZE,
                 max_output_size=MAX_OUTPUT_SIZE,
                 iou_threshold=iou_threshold,
                 confidence_threshold=confidence_threshold)

    if type == 'images':
        batch_size = len(input_names)
        batch = utility_functions.load_images(input_names, model_size=MODEL_SIZE)
        inputs = tf.placeholder(tf.float32, [batch_size, *MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.train.Saver(tf.global_variables(scope='yolo_model'))

        with tf.Session() as sess2:
            saver.restore(sess2, './data/model.ckpt')
            print("Model restored successfully.")
            detection_result = sess2.run(detections, feed_dict={inputs: batch})


        utility_functions.draw_boxes(input_names, detection_result, class_names, MODEL_SIZE)

        print('Detections have been saved successfully.')

    elif type == 'video':
        inputs = tf.placeholder(tf.float32, [1, *MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.train.Saver(tf.global_variables(scope='yolo_model'))

        with tf.Session() as sess2:
            saver.restore(sess2, './data/model.ckpt')

            win_name = 'Video detection'
            cv2.namedWindow(win_name)
            cap = cv2.VideoCapture(input_names[0])
            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('./detections/detections.mp4', fourcc, fps,
                                  (int(frame_size[0]), int(frame_size[1])))

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized_frame = cv2.resize(frame, dsize=MODEL_SIZE[::-1],
                                               interpolation=cv2.INTER_NEAREST)
                    detection_result = sess2.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    utility_functions.draw_frame(frame, frame_size, detection_result,
                                                 class_names, MODEL_SIZE)

                    cv2.imshow(win_name, frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    out.write(frame)
            finally:
                cv2.destroyAllWindows()
                cap.release()
                print("Detections have been saved successfully.")

    else:
        raise ValueError("Inappropriate data type. Please choose either 'video' or 'images'.")


def run():
    if not(os.path.isfile(MODEL_FILE)):
        print("Trying to build model.")
        if not(load_weights.build()):
            print("Error with weigths file.")
            return
        print("Please restart.")
        return

    iou_threshold = 0.5
    confidence_threshold = 0.5
    files = []
    for file in os.listdir(IMAGES_DIR):
        if file.endswith('.jpg'):
            image_file = os.path.join(IMAGES_DIR, file)
            image_file = os.path.abspath(image_file)
            files.append(image_file)

    main("images", iou_threshold, confidence_threshold, files)


if __name__ == '__main__':
    run()
