"""Yolo

"""

import tensorflow as tf
import sys
import cv2

from yolo import Yolo
from utility_functions import load_images, load_class_names, draw_boxes, draw_frame

MODEL_SIZE = (416, 416)
CLASS_NAMES_FILE = './data/labels/coco.names'
MAX_OUTPUT_SIZE = 20


def main(type, iou_threshold, confidence_threshold, input_names):
    class_names = load_class_names(CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo(n_classes=n_classes, model_size=MODEL_SIZE,
                    max_output_size=MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

    if type == 'images':
        batch_size = len(input_names)
        batch = load_images(input_names, model_size=MODEL_SIZE)
        inputs = tf.placeholder(tf.float32, [batch_size, *MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

        with tf.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')
            detection_result = sess.run(detections, feed_dict={inputs: batch})

        draw_boxes(input_names, detection_result, class_names, MODEL_SIZE)

        print('Detections have been saved successfully.')

    elif type == 'video':
        inputs = tf.placeholder(tf.float32, [1, *MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

        with tf.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')

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
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    draw_frame(frame, frame_size, detection_result,
                               class_names, MODEL_SIZE)

                    cv2.imshow(win_name, frame)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    out.write(frame)
            finally:
                cv2.destroyAllWindows()
                cap.release()
                print('Detections have been saved successfully.')

    else:
        raise ValueError("Inappropriate data type. Please choose either 'video' or 'images'.")


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
