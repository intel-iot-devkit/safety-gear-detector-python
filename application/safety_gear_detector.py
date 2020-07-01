#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function
import sys
import os
import cv2
import numpy as np
from argparse import ArgumentParser
import datetime
import json
from inference import Network

# Global vars
cpu_extension = ''
conf_modelLayers = ''
conf_modelWeights = ''
conf_safety_modelLayers = ''
conf_safety_modelWeights = ''
targetDevice = "CPU"
conf_batchSize = 1
conf_modelPersonLabel = 1
conf_inferConfidenceThreshold = 0.7
conf_inFrameViolationsThreshold = 19
conf_inFramePeopleThreshold = 5
use_safety_model = False
padding = 30
viol_wk = 0
acceptedDevices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HDDL']
videos = []
name_of_videos = []
CONFIG_FILE = '../resources/config.json'
is_async_mode = True


class Video:
    def __init__(self, idx, path):
        if path.isnumeric():
            self.video = cv2.VideoCapture(int(path))
            self.name = "Cam " + str(idx)
        else:
            if os.path.exists(path):
                self.video = cv2.VideoCapture(path)
                self.name = "Video " + str(idx)
            else:
                print("Either wrong input path or empty line is found. Please check the conf.json file")
                exit(21)
        if not self.video.isOpened():
            print("Couldn't open video: " + path)
            sys.exit(20)
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.currentViolationCount = 0
        self.currentViolationCountConfidence = 0
        self.prevViolationCount = 0
        self.totalViolations = 0
        self.totalPeopleCount = 0
        self.currentPeopleCount = 0
        self.currentPeopleCountConfidence = 0
        self.prevPeopleCount = 0
        self.currentTotalPeopleCount = 0

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.frame_start_time = datetime.datetime.now()


def get_args():
    """
    Parses the argument.
    :return: None
    """
    global is_async_mode
    parser = ArgumentParser()
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU,"
                             "FPGA, MYRIAD or HDDL is acceptable. Application will"
                             "look for a suitable plugin for device specified"
                             " (CPU by default)",
                        type=str, required=False)
    parser.add_argument("-m", "--model",
                        help="Path to an .xml file with a trained model's"
                             " weights.",
                        required=True, type=str)
    parser.add_argument("-sm", "--safety_model",
                        help="Path to an .xml file with a trained model's"
                             " weights.",
                        required=False, type=str, default=None)
    parser.add_argument("-e", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                             "path to a shared library with the kernels impl",
                        type=str, default=None)
    parser.add_argument("-f", "--flag", help="sync or async", default="async", type=str)

    args = parser.parse_args()

    global conf_modelLayers, conf_modelWeights, conf_safety_modelLayers, conf_safety_modelWeights, \
           targetDevice, cpu_extension, videos, use_safety_model
    if args.model:
        conf_modelLayers = args.model
        conf_modelWeights = os.path.splitext(conf_modelLayers)[0] + ".bin"
    if args.safety_model:
        conf_safety_modelLayers = args.safety_model
        conf_safety_modelWeights = os.path.splitext(conf_safety_modelLayers)[0] + ".bin"  
        use_safety_model = True      
    if args.device:
        targetDevice = args.device
        if "MULTI:" not in targetDevice:
           if targetDevice not in acceptedDevices:
               print("Selected device, %s not supported." % (targetDevice))
               sys.exit(12)
    if args.cpu_extension:
        cpu_extension = args.cpu_extension
    if args.flag == "async":
        is_async_mode = True
        print('Application running in Async mode')
    else:
        is_async_mode = False
        print('Application running in Sync mode')
    assert os.path.isfile(CONFIG_FILE), "{} file doesn't exist".format(CONFIG_FILE)
    config = json.loads(open(CONFIG_FILE).read())
    for idx, item in enumerate(config['inputs']):
        vid = Video(idx, item['video'])
        name_of_videos.append([idx, item['video']])
        videos.append([idx, vid])


def detect_safety_hat(img):
    """
    Detection of the hat of the person.
    :param img: Current frame
    :return: Boolean value of the detected hat
    """
    lowH = 15
    lowS = 65
    lowV = 75

    highH = 30
    highS = 255
    highV = 255

    crop = 0
    height = 15
    perc = 8

    hsv = np.zeros(1)

    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        print("%d %d %d" % (img.shape))
        print("%d %d %d" % (img.shape))
        print(e)

    threshold_img = cv2.inRange(hsv, (lowH, lowS, lowV), (highH, highS, highV))

    x = 0
    y = int(threshold_img.shape[0] * crop / 100)
    w = int(threshold_img.shape[1])
    h = int(threshold_img.shape[0] * height / 100)
    img_cropped = threshold_img[y: y + h, x: x + w]

    if cv2.countNonZero(threshold_img) < img_cropped.size * perc / 100:
        return False
    return True


def detect_safety_jacket(img):
    """
    Detection of the safety jacket of the person.
    :param img: Current frame
    :return: Boolean value of the detected jacket
    """
    lowH = 0
    lowS = 150
    lowV = 42

    highH = 11
    highS = 255
    highV = 255

    crop = 15
    height = 40
    perc = 23

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    threshold_img = cv2.inRange(hsv, (lowH, lowS, lowV), (highH, highS, highV))

    x = 0
    y = int(threshold_img.shape[0] * crop / 100)
    w = int(threshold_img.shape[1])
    h = int(threshold_img.shape[0] * height / 100)
    img_cropped = threshold_img[y: y + h, x: x + w]

    if cv2.countNonZero(threshold_img) < img_cropped.size * perc / 100:
        return False
    return True


def detect_workers(workers, frame):
    """
    Detection of the person with the safety guards.
    :param workers: Total number of the person in the current frame
    :param frame: Current frame
    :return: Total violation count of the person
    """
    violations = 0
    global viol_wk
    for worker in workers:
        xmin, ymin, xmax, ymax = worker
        crop = frame[ymin:ymax, xmin:xmax]
        if 0 not in crop.shape:
            if detect_safety_hat(crop):
                if detect_safety_jacket(crop):
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                  (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                  (0, 0, 255), 2)
                    violations += 1
                    viol_wk += 1

            else:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                violations += 1
                viol_wk += 1
    return violations


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    get_args()
    global is_async_mode
    nextReq = 1
    currReq = 0
    nextReq_s = 1
    currReq_s = 0
    prevVideo = None
    vid_finished = [False] * len(videos)
    min_FPS = min([videos[i][1].video.get(cv2.CAP_PROP_FPS) for i in range(len(videos))])

    # Initialise the class
    infer_network = Network()
    infer_network_safety = Network()
    # Load the network to IE plugin to get shape of input layer
    plugin, (batch_size, channels, model_height, model_width) = \
        infer_network.load_model(conf_modelLayers, targetDevice, 1, 1, 2, cpu_extension)
    if use_safety_model:
        batch_size_sm, channels_sm, model_height_sm, model_width_sm = \
            infer_network_safety.load_model(conf_safety_modelLayers, targetDevice, 1, 1, 2, cpu_extension, plugin)[1]

    while True:
        for index, currVideo in videos:
            # Read image from video/cam
            vfps = int(round(currVideo.video.get(cv2.CAP_PROP_FPS)))
            for i in range(0, int(round(vfps / min_FPS))):
                ret, current_img = currVideo.video.read()
                if not ret:
                    vid_finished[index] = True
                    break
            if vid_finished[index]:
                stream_end_frame = np.zeros((int(currVideo.height), int(currVideo.width), 1),
                                               dtype='uint8')
                cv2.putText(stream_end_frame, "Input file {} has ended".format
                (name_of_videos[index][1].split('/')[-1]) ,
                            (10, int(currVideo.height/2)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(currVideo.name, stream_end_frame)
                continue
            # Transform image to person detection model input
            rsImg = cv2.resize(current_img, (model_width, model_height))
            rsImg = rsImg.transpose((2, 0, 1))
            rsImg = rsImg.reshape((batch_size, channels, model_height, model_width))

            infer_start_time = datetime.datetime.now()
            # Infer current image
            if is_async_mode:
                infer_network.exec_net(nextReq, rsImg)
            else:
                infer_network.exec_net(currReq, rsImg)
                prevVideo = currVideo
                previous_img = current_img

            # Wait for previous request to end
            if infer_network.wait(currReq) == 0:
                infer_end_time = (datetime.datetime.now() - infer_start_time) * 1000

                in_frame_workers = []

                people = 0
                violations = 0
                hard_hat_detection =False
                vest_detection = False
                result = infer_network.get_output(currReq)
                # Filter output
                for obj in result[0][0]:
                    if obj[2] > conf_inferConfidenceThreshold:
                        xmin = int(obj[3] * prevVideo.width)
                        ymin = int(obj[4] * prevVideo.height)
                        xmax = int(obj[5] * prevVideo.width)
                        ymax = int(obj[6] * prevVideo.height)
                        xmin = int(xmin - padding) if (xmin - padding) > 0 else 0
                        ymin = int(ymin - padding) if (ymin - padding) > 0 else 0
                        xmax = int(xmax + padding) if (xmax + padding) <  prevVideo.width else  prevVideo.width
                        ymax = int(ymax + padding) if (ymax + padding) <  prevVideo.height else  prevVideo.height
                        cv2.rectangle(previous_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        people += 1
                        in_frame_workers.append((xmin, ymin, xmax, ymax))
                        new_frame = previous_img[ymin:ymax, xmin:xmax]
                        if use_safety_model:
                            
                            # Transform image to safety model input
                            in_frame_sm = cv2.resize(new_frame, (model_width_sm, model_height_sm))
                            in_frame_sm = in_frame_sm.transpose((2, 0, 1))
                            in_frame_sm = in_frame_sm.reshape((batch_size_sm, channels_sm, model_height_sm, model_width_sm))

                            infer_start_time_sm = datetime.datetime.now()
                            if is_async_mode:
                                infer_network_safety.exec_net(nextReq_s, in_frame_sm)
                            else:
                                infer_network_safety.exec_net(currReq_s, in_frame_sm)
                            # Wait for the result
                            infer_network_safety.wait(currReq_s)
                            infer_end_time_sm = (datetime.datetime.now() - infer_start_time_sm) * 1000

                            result_sm = infer_network_safety.get_output(currReq_s)
                            # Filter output
                            hard_hat_detection = False
                            vest_detection = False
                            detection_list = []
                            for obj_sm in result_sm[0][0]:

                                if (obj_sm[2] > 0.4):
                                    # Detect safety vest
                                    if (int(obj_sm[1])) == 2:
                                        xmin_sm = int(obj_sm[3] * (xmax-xmin))
                                        ymin_sm = int(obj_sm[4] * (ymax-ymin))
                                        xmax_sm = int(obj_sm[5] * (xmax-xmin))
                                        ymax_sm = int(obj_sm[6] * (ymax-ymin))
                                        if vest_detection == False:
                                            detection_list.append([xmin_sm+xmin, ymin_sm+ymin, xmax_sm+xmin, ymax_sm+ymin])
                                            vest_detection = True

                                    # Detect hard-hat
                                    if int(obj_sm[1]) == 4:
                                        xmin_sm_v = int(obj_sm[3] * (xmax-xmin))
                                        ymin_sm_v = int(obj_sm[4] * (ymax-ymin))
                                        xmax_sm_v = int(obj_sm[5] * (xmax-xmin))
                                        ymax_sm_v = int(obj_sm[6] * (ymax-ymin))
                                        if hard_hat_detection == False:
                                            detection_list.append([xmin_sm_v+xmin, ymin_sm_v+ymin, xmax_sm_v+xmin, ymax_sm_v+ymin])
                                            hard_hat_detection = True

                            if hard_hat_detection is False or vest_detection is False:
                                violations += 1
                            for _rect in detection_list:
                                cv2.rectangle(current_img, (_rect[0] , _rect[1]), (_rect[2] , _rect[3]), (0, 255, 0), 2)
                            if is_async_mode:
                                currReq_s, nextReq_s = nextReq_s, currReq_s

                    # Use OpenCV if worker-safety-model is not provided
                        else :
                            violations = detect_workers(in_frame_workers, previous_img)

                # Check if detected violations equals previous frames
                if violations == prevVideo.currentViolationCount:
                    prevVideo.currentViolationCountConfidence += 1

                    # If frame threshold is reached, change validated count
                    if prevVideo.currentViolationCountConfidence == conf_inFrameViolationsThreshold:

                        # If another violation occurred, save image
                        if prevVideo.currentViolationCount > prevVideo.prevViolationCount:
                            prevVideo.totalViolations += (prevVideo.currentViolationCount - prevVideo.prevViolationCount)
                        prevVideo.prevViolationCount = prevVideo.currentViolationCount
                else:
                    prevVideo.currentViolationCountConfidence = 0
                    prevVideo.currentViolationCount = violations

                # Check if detected people count equals previous frames
                if people == prevVideo.currentPeopleCount:
                    prevVideo.currentPeopleCountConfidence += 1

                    # If frame threshold is reached, change validated count
                    if prevVideo.currentPeopleCountConfidence == conf_inFrameViolationsThreshold:
                        prevVideo.currentTotalPeopleCount += (
                                prevVideo.currentPeopleCount - prevVideo.prevPeopleCount)
                        if prevVideo.currentTotalPeopleCount > prevVideo.prevPeopleCount:
                            prevVideo.totalPeopleCount += prevVideo.currentTotalPeopleCount - prevVideo.prevPeopleCount
                        prevVideo.prevPeopleCount = prevVideo.currentPeopleCount
                else:
                    prevVideo.currentPeopleCountConfidence = 0
                    prevVideo.currentPeopleCount = people



                frame_end_time = datetime.datetime.now()
                cv2.putText(previous_img, 'Total people count: ' + str(
                    prevVideo.totalPeopleCount), (10, prevVideo.height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(previous_img, 'Current people count: ' + str(
                    prevVideo.currentTotalPeopleCount),
                            (10, prevVideo.height - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(previous_img, 'Total violation count: ' + str(
                    prevVideo.totalViolations), (10, prevVideo.height - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(previous_img, 'FPS: %0.2fs' % (1 / (
                        frame_end_time - prevVideo.frame_start_time).total_seconds()),
                            (10, prevVideo.height - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(previous_img, "Inference time: N\A for async mode" if is_async_mode else\
				"Inference time: {:.3f} ms".format((infer_end_time).total_seconds()),
                            (10, prevVideo.height - 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow(prevVideo.name, previous_img)
                prevVideo.frame_start_time = datetime.datetime.now()
            # Swap
            if is_async_mode:
                currReq, nextReq = nextReq, currReq
                previous_img = current_img
                prevVideo = currVideo
            if cv2.waitKey(1) == 27:
                print("Attempting to stop input files")
                infer_network.clean()
                infer_network_safety.clean()
                cv2.destroyAllWindows()
                return

        if False not in vid_finished:
            infer_network.clean()
            infer_network_safety.clean()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
