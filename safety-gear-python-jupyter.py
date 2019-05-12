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
import datetime

from inference import Network

# Global vars
cpu_extension = ''
conf_modelLayers = ''
conf_modelWeights = ''
targetDevice = "CPU"
conf_batchSize = 1
conf_modelPersonLabel = 1
conf_inferConfidenceThreshold = 0.7
conf_inFrameViolationsThreshold = 15
conf_inFramePeopleThreshold = 5
padding = 0.05
viol_wk = 0
acceptedDevices = ['CPU', 'GPU', 'MYRIAD', 'HETERO:FPGA,CPU', 'HETERO:HDDL,CPU']
videos = []
name_of_videos = []


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
                print("Either wrong input path or empty line is found. Please check the conf.txt file")
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


def env_parser():
    """
    Parses the inputs.
    :return: None
    """
    global conf_modelLayers, conf_modelWeights, targetDevice, cpu_extension, videos
    if 'MODEL' in os.environ:
        conf_modelLayers = os.environ['MODEL']
        conf_modelWeights = os.path.splitext(conf_modelLayers)[0] + ".bin"
    else:
        print("Please provide path for the .xml file.")
        sys.exit(0)
    if 'DEVICE' in os.environ:
        targetDevice = os.environ['DEVICE']
        if targetDevice not in acceptedDevices:
            print("Selected device, %s not supported." % (targetDevice))
            sys.exit(12)
    if 'CPU_EXTENSION' in os.environ:
        cpu_extension = os.environ['CPU_EXTENSION']
    if 'CONFIG' in os.environ:
        with open(os.environ['CONFIG'], 'r') as cfg:
            for cnt, line in enumerate(cfg.read().splitlines()):
                vid = Video(cnt, line)
                name_of_videos.append([cnt, line])
                videos.append([cnt, vid])
    else:
        print("Please provide path for the conf.txt")
        sys.exit(0)


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
    env_parser()
    prevReq = 0
    currReq = 1

    prevVideo = None
    vid_finished = [False] * len(videos)
    min_FPS = min([videos[i][1].video.get(cv2.CAP_PROP_FPS) for i in range(len(videos))])
    wait_time = int(round(1000 / min_FPS / len(videos)))

    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    batch_size, channels, model_height, model_width = \
        infer_network.load_model(conf_modelLayers, targetDevice, 1, 1, 2,
                                 cpu_extension)[1]

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
            # Transform image to model input
            rsImg = cv2.resize(current_img, (model_width, model_height))
            rsImg = rsImg.transpose((2, 0, 1))
            rsImg = rsImg.reshape(
                (batch_size, channels, model_height, model_width))

            infer_start_time = datetime.datetime.now()
            # Infer current image
            infer_network.exec_net(currReq, rsImg)

            # Wait for previous request to end
            if infer_network.wait(prevReq) == 0:
                infer_end_time = (datetime.datetime.now() - infer_start_time) * 1000

                in_frame_workers = []

                people = 0
                result = infer_network.get_output(prevReq)
                # Filter output
                for obj in result[0][0]:
                    if obj[2] > conf_inferConfidenceThreshold:
                        xmin = int(obj[3] * prevVideo.width)
                        ymin = int(obj[4] * prevVideo.height)
                        xmax = int(obj[5] * prevVideo.width)
                        ymax = int(obj[6] * prevVideo.height)

                        ymin = ymin - int(padding * (ymax - ymin))
                        in_frame_workers.append((xmin, ymin, xmax, ymax))
                        people += 1

                violations = detect_workers(in_frame_workers, previous_img)
                # Check if detected violations equals previous frames
                if violations == prevVideo.currentViolationCount:
                    prevVideo.currentViolationCountConfidence += 1
                    # If frame threshold is reached, change validated count
                    if prevVideo.currentViolationCountConfidence == conf_inFrameViolationsThreshold:
                        # If another violation occurred, save image
                        if prevVideo.currentViolationCount > prevVideo.prevViolationCount:
                            prevVideo.totalViolations += (
                                    prevVideo.currentViolationCount - prevVideo.prevViolationCount)
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
                cv2.putText(previous_img, 'Inference time: {}ms'.format((infer_end_time).total_seconds()),
                            (10, prevVideo.height - 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(prevVideo.name, previous_img)
                prevVideo.frame_start_time = datetime.datetime.now()

            # Swap
            currReq, prevReq = prevReq, currReq
            previous_img = current_img
            prevVideo = currVideo
        # Exit if ESC key is pressed
        if cv2.waitKey(wait_time) == 27:
            print("Attempting to stop input files")
            break
        if False not in vid_finished:
            break
    infer_network.clean()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
