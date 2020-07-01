#!/bin/bash
# Copyright (c) 2018 Intel Corporation.
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

BASE_DIR=`pwd`

#Install the dependencies
sudo apt-get update
sudo apt-get install ffmpeg
sudo apt-get install python3-pip
sudo pip3 install numpy jupyter

#Download the person detection model
cd /opt/intel/openvino/deployment_tools/tools/model_downloader
sudo ./downloader.py --name person-detection-retail-0013

#Optimize the worker-safety-mobilenet model
cd /opt/intel/openvino/deployment_tools/model_optimizer/
./mo_caffe.py --input_model $BASE_DIR/resources/worker-safety-mobilenet/worker_safety_mobilenet.caffemodel  -o $BASE_DIR/resources/worker-safety-mobilenet/FP32 --data_type FP32
./mo_caffe.py --input_model $BASE_DIR/resources/worker-safety-mobilenet/worker_safety_mobilenet.caffemodel  -o $BASE_DIR/resources/worker-safety-mobilenet/FP16 --data_type FP16

