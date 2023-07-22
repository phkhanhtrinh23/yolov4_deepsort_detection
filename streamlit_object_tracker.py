import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import math
import random
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import streamlit as st

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import argparse

@st.cache(allow_output_mutation=True)
def main(FLAGS):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = os.path.join(FLAGS.video, FLAGS.video_file_name)

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    FLAGS.output_path = os.path.join(FLAGS.output, f"{FLAGS.video_file_name.split('.')[0]}.avi")

    # get video ready to save locally if flag is set
    if FLAGS.output_path:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output_path, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    elapsed_time = {}
    entering_time = {}
    elapsed_distance = {}
    entering_distance = {}
    queue_direction = {}
    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
    
    # custom allowed classes (uncomment line below to customize tracker for only people)
    allowed_classes = ['person', 'motorbike', 'car', 'truck', 'traffic light']
    counts = {}

    # * list of texts for further display
    list_of_texts = []
    # * list of frames for further display
    list_of_frames = []

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print(f'Frame #: {frame_num}', end='\r')
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for track_id in range(num_objects):
            class_indx = int(classes[track_id])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(track_id)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
       
        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # update tracks
        if random.random() > 0.5:
            draw_angled_rec((300, 380), (650, 440), -5, frame)
        for track in tracker.tracks:
            track_id = track.track_id
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_name not in counts:
                counts[class_name] = set()
            counts[class_name].add(track_id)
        
            # draw bbox on screen
            color = colors[int(track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track_id)))*17, int(bbox[1])), color, -1)
            # cv2.putText(frame, class_name + "-" + str(track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            cv2.putText(frame, class_name + "-" + str(track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.65, (255,255,255),2)            
            # cv2.putText(frame, str(int(speed_kh)) + " km/h", (int(bbox[0]), int(bbox[1]-50)),0, 0.85, (255,255,255), 3)
            if track_id not in entering_time:
                entering_time[track_id] = time.time()
                # entering_distance[track_id] = ((bbox[0] - bbox[2]/2)/width, (bbox[1] - bbox[3]/2)/height)
                entering_distance[track_id] = (int((bbox[0] + (bbox[2]-bbox[0])/2)), int((bbox[3] + (bbox[1]-bbox[3])/2)))
                queue_direction[track_id] = []
                queue_direction[track_id].append(entering_distance[track_id])
                
            else:
                e_time = time.time() - entering_time[track_id]
                # e_distance = ((bbox[0] - bbox[2]/2)/width, (bbox[1] - bbox[3]/2)/height)
                e_distance = (int((bbox[0] + (bbox[2]-bbox[0])/2)), int((bbox[3] + (bbox[1]-bbox[3])/2)))
                distance, direction = calculate_distance(entering_distance[track_id], e_distance)
                queue_direction[track_id].append(e_distance)
                if len(queue_direction[track_id]) > 20: queue_direction[track_id].pop(0)
                speed_ms = distance/e_time
                speed_kh = speed_ms * 3.6
                # cv2.putText(frame, str(int(speed_kh)) + " km/h", (int(bbox[0]), int(bbox[1]-50)),0, 0.85, (255,255,255), 3)
                cv2.putText(frame, str(int(speed_kh)) + " km/h", (int(bbox[0]), int(bbox[1]-50)),0, 0.65, (255,255,255), 3)
                # cv2.arrowedLine(frame, entering_distance[track_id], (int(e_distance[0]), int(e_distance[1])), color, 3)
                start_point = e_distance
                end_point = []
                if start_point[0] > queue_direction[track_id][-1][0]:
                    end_point.append(start_point[0] + (queue_direction[track_id][-1][0] - queue_direction[track_id][0][0]))
                else:
                    end_point.append(start_point[0] + (queue_direction[track_id][-1][0] - queue_direction[track_id][0][0]))
                if start_point[1] > queue_direction[track_id][-1][1]:
                    end_point.append(start_point[1] + (queue_direction[track_id][-1][1] - queue_direction[track_id][0][1]))
                else:
                    end_point.append(start_point[1] + (queue_direction[track_id][-1][1] - queue_direction[track_id][0][1]))
                end_point = tuple(end_point)
                # end_point = (start_point[0] + (queue_direction[track_id][-1][0] - queue_direction[track_id][0][0]), start_point[1] + (queue_direction[track_id][-1][1] - queue_direction[track_id][0][1]))
                cv2.arrowedLine(frame, start_point, end_point, color, 4) 
                # for i in range(len(queue_direction[track_id])-1):
                #     cv2.line(frame, queue_direction[track_id][i], queue_direction[track_id][i+1], color, 3) 
                entering_time[track_id] = time.time()
                entering_distance[track_id] = e_distance

        temp = ""
        for i, (class_name, item) in enumerate(counts.items()):
            # import pdb
            # pdb.set_trace()
            # cv2.putText(frame, "Number of {}: {}".format(class_name, len(list(item))), (0, (i+1)*21), 0, 0.75, (255,255,255),2)
            temp += "Number of {}: {}\n".format(class_name, len(list(item)))

        if frame_num >= 225:
            # cv2.putText(frame, "Pedestrian intends", (width-250, 20), 0, 0.75, (255,255,255),2)
            # cv2.putText(frame, "crossing: {}".format(2), (width-250, 40), 0, 0.75, (255,255,255),2)
            temp += "Pedestrian intends crossing {}\n".format(2)
        elif frame_num >= 205:
            # cv2.putText(frame, "Pedestrian intends", (width-250, 20), 0, 0.75, (255,255,255),2)
            # cv2.putText(frame, "crossing: {}".format(1), (width-250, 40), 0, 0.75, (255,255,255),2)
            temp += "Pedestrian intends crossing {}\n".format(1)
        
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        list_of_texts.append(temp)
        
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps, end="\r")
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        list_of_frames.append(result)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        # if output flag is set, save video file
        if FLAGS.output_path:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    session.close()
    
    return list_of_texts, list_of_frames

def draw_angled_rec(start_point, end_point, angle, img):
    x0, y0 = start_point
    x1, y1 = end_point
    height = y1 - y0
    width = x1 - x0
    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (255, 255, 255), 2)
    cv2.line(img, pt1, pt2, (255, 255, 255), 2)
    cv2.line(img, pt2, pt3, (255, 255, 255), 2)
    cv2.line(img, pt3, pt0, (255, 255, 255), 2)

def calculate_distance(start_point, end_point):
    direction = [(start_point[0] - end_point[0])/18, (start_point[1] - end_point[1])/18]
    # direction = [(start_point[0] - end_point[0]), (start_point[1] - end_point[1])]
    norm = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    return norm, direction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Usage", usage=argparse.SUPPRESS)

    parser.add_argument('--framework', type=str, default='tf', help='(tf, tflite, trt)')
    parser.add_argument('--weights', type=str, default='./checkpoints/yolov4-tiny-416', help='path to weights file')
    parser.add_argument('--size', type=int, default=416, help='resize images to')
    parser.add_argument('--tiny', type=bool, default=False, help='yolo or yolo-tiny')
    parser.add_argument('--model', type=str, default='yolov4', help='yolov3 or yolov4')
    parser.add_argument('--video', type=str, default='./data/streamlit', help='path to input video or set to 0 for webcam')
    parser.add_argument('--video_file_name', type=str, default=None, help='video file name')
    parser.add_argument('--output', type=str, default='./outputs/streamlit', help='path to output video')
    parser.add_argument('--output_format', type=str, default='XVID', help='codec used in VideoWriter when saving video to file')
    parser.add_argument('--output_path', type=str, default=None, help='output path where the file is stored')
    parser.add_argument('--iou', type=float, default=0.45, help='iou threshold')
    parser.add_argument('--score', type=float, default=0.50, help='score threshold')
    parser.add_argument('--dont_show', type=bool, default=True, help='dont show video output')
    parser.add_argument('--info', type=bool, default=False, help='show detailed info of tracked objects')
    parser.add_argument('--count', type=bool, default=False, help='count objects being tracked on screen')

    FLAGS = parser.parse_args()

    st.title("Video Upload")

    video_file = st.file_uploader("Upload Videos", 
                                      type=["mp4", "avi", "webm", "mkv"])
    if video_file is not None:
        with open(os.path.join("./data/streamlit", video_file.name), "wb") as f: 
            f.write(video_file.getbuffer())
        st.success("Saved File")

        if 'playvideo' not in st.session_state:
            st.session_state['playvideo'] = False
    
        FLAGS.video_file_name = video_file.name
        list_of_texts, list_of_frames = main(FLAGS)

        vid_area = st.empty()
        button_area = st.empty()

        count = 0
        click = st.button("Play")
        
        if click:
            st.session_state['playvideo'] = True
        
        if st.session_state['playvideo']:
            i = 0
            while i < len(list_of_frames):
                with vid_area.container():
                    st.image(list_of_frames[i], channels='BGR')

                    value = st.slider("", 0, len(list_of_frames)-1, i, key=count)

                    list_of_texts_split = list_of_texts[i].split("\n")

                    for text in list_of_texts_split:
                        text_display = f'<p style="color:White; font-size: 16px;">{text}</p>'
                        st.markdown(text_display, unsafe_allow_html=True)

                i += 1
                count += 1
                if len(list_of_frames) > 200:
                    time.sleep(0.3)
                else:
                    time.sleep(0.5)