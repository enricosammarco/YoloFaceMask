import tensorflow as tf
from yolo import YOLO
from yolo3.model import *
from yolo3.utils import *
import cv2
from imutils.video import FPS
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def detecting_image(detector, image):
    bb = []
    classi = []
    if detector.model_image_size != (None, None):
        assert detector.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert detector.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(detector.model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    # print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = detector.sess.run(
        [detector.boxes, detector.scores, detector.classes],
        feed_dict={
            detector.yolo_model.input: image_data,
            detector.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = detector.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        # print(label, (left, top), (right, bottom))

        bb.append([left, top, right, bottom])
        if predicted_class == "face":
            classi.append(0)
        else:
            classi.append(1)

    bb_final = []
    classes_final = []
    for index in range(0,len(classi)):
        if classi[index] == 1:
            bb_final.append(bb[index])
            classes_final.append(classi[index])

    bb_0 = []
    for index in range(0, len(classi)):
        if classi[index] == 0:
            flag = True
            for mask in bb_final:
                iou = bb_intersection_over_union(mask,bb[index])
                if iou>0.5:
                    flag = False
                    break
            if flag:
                bb_0.append(bb[index])
    for i in bb_0:
        bb_final.append(i)
        classes_final.append(0)

    return bb_final, classes_final, image


model_path = "weights_and_anchors/yolov3ep054.h5"

arg = {
      "model_path": model_path,
      "anchors_path": 'weights_and_anchors/yolo_anchors.txt', #YOLOV3
      "classes_path": 'weights_and_anchors/classes.txt',
      "score" : 0.3,
      "iou" : 0.45,
      "model_image_size" : (416, 416), #YOLOV3
      "gpu_num" : 1,
  }
detector = YOLO(**arg)

cap = cv2.VideoCapture(0)
framen = 0


while True:
    # Read frame
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break
    fps = FPS().start()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # for detected in output_info:
    #     rect = dlib.rectangle(detected[2], detected[3], detected[4], detected[5])
    #     shape = predictor(img, rect)
    #
    #     c = (0, 0, 255)
    #     for i in range(0, 68):
    #         p = (shape.part(i).x, shape.part(i).y)
    #         cv2.circle(image, p, 2, c, -1)

    predicted, pred_classes, image = detecting_image(detector, img)

    for bb in range(0,len(predicted)):
        if pred_classes[bb] == 0:
            frame = cv2.rectangle(frame,(predicted[bb][0], predicted[bb][1]),(predicted[bb][2],predicted[bb][3]),(0,0,255),5)
        else:
            frame = cv2.rectangle(frame, (predicted[bb][0], predicted[bb][1]), (predicted[bb][2], predicted[bb][3]),(0, 255,0), 5)

    fps.update()
    fps.stop()

    info = []
    info.append("Face + Mask: green bounding box")
    info.append("Face: red bounding box")
    info.append("Number of detection: "+str(len(predicted)))
    info.append("FPS "+"{:.2f}".format(fps.fps()))

    j = 1
    for text in info:
        cv2.putText(frame, text, (0, frame.shape[0] - (( j * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        j += 1


    # image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("", frame)
    k = cv2.waitKey(40) & 0xff
    if k == 27:
        break
