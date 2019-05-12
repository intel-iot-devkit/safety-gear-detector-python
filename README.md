# Safety Gear Detector

| Details           |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  Python\* 3.5|
| Time to Complete:    |  30-40min     |

![safety-gear-detector](docs/images/safetygear.png)

Figure 1: An application capable of detecting people and if they are wearing safety-jackets and hard-hats in a video. 

## What It Does
This application is one of a series of IoT reference implementations illustrating how to develop a working solution for a particular problem. It demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. 
This reference implementation detects people and potential violations of safety-gear standards.

## How It Works
The application uses the Inference Engine included in Intel® Distribution of OpenVINO™ toolkit. A trained neural network detects people within a designated area. The application checks if each detected person is wearing a safety-jacket and hard-hat. If they are not, a red bounding box is drawn over the person in the video. If they are wearing their designated safety gear, a green bounding box is drawn over them in the video.

![Architectural diagram](docs/images/archdia.png)
## Requirements
### Hardware
* 6th to 8th Generation Intel® Core™ processor with Intel® Iris® Pro graphics or Intel® HD Graphics

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)

   **Note:** We recommend using a 4.14+ Linux kernel with this software. Run the following command to determine your kernel version:

   ```
   uname -a
   ```
* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit 2019 R1 release

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit
Refer to [Install the Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) to install and set up the toolkit.

Install the OpenCL™ Runtime Package to run inference on the GPU. It is not mandatory for CPU inference.

### FFmpeg*
FFmpeg is installed separately from the Ubuntu repositories:
```
sudo apt update
sudo apt install ffmpeg
```

### Install Python* dependencies
```
sudo apt install python3-pip

sudo pip3 install numpy

sudo pip3 install jupyter
```

## Configure the Application

### Which Model to Use
By default, this application uses the **person-detection-retail-0013** Intel® model, that can be accessed using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that will be used by the application.

#### Download the __.xml__ and __.bin__ files

Go to the **model downloader** directory present inside Intel® Distribution of OpenVINO™ toolkit.

  ```
  cd /opt/intel/openvino/deployment_tools/tools/model_downloader
  ```

- Specify which model to download with `--name`. <br><br>
- To optimize the model for FP32, run the following command:

  ```
  sudo ./downloader.py --name person-detection-retail-0013
  ```
- To optimize the model for FP16, run the following command:
   ```
   sudo ./downloader.py --name person-detection-retail-0013-fp16
   ```
The files will be downloaded inside the Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt directory.  
### The Config File
The _resources/conf.txt_ contains the videos that will be used by the application, one video per line.   
Each of the lines in the file is of the form `path/to/video`.

For example:
```
videos/video1.mp4
```
The `path/to/video` is the path, on the local system, to a video to use as input.

The application can use any number of videos for detection (i.e., The _conf.txt_ file can have any number of lines.), but the more videos the application uses in parallel, the more the frame rate of each video scales down. This can be solved by adding more computation power to the machine on which the application is running.
   

### What Input Video to Use
The application works with any input video. We recommend using the [Safety_Full_Hat_and_Vest.mp4](resources/Safety_Full_Hat_and_Vest.mp4) video.   

<!-- This video can be downloaded directly, via the `video_downloader` python script provided. The script works with both python2 and python3. Run the following command:
```
python video_downloader.py
```
The video is automatically downloaded to the `resources/` folder. -->

### Use a Camera Stream
Replace `path/to/video` with the camera ID in conf.txt, where the ID is taken from your video device (the number X in /dev/videoX).
On Ubuntu, to list all available video devices use the following command:
  ```
  ls /dev/video*
  ```
For example, if the output of above command is `/dev/video0`, then conf.txt would be:
```
0
```

## Set Up the Environment

Open the terminal to setup the environment variables required to run the Intel® Distribution of OpenVINO™ toolkit applications:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```
**Note:** This command only needs to be executed once in the terminal where the application will be executed. If the terminal is closed, the command needs to be executed again.   


## Run the Application
Change the current directory to the git-cloned application code location on your system. For example:
```
cd <path_to_the_safety-gear-detector-python_directory>
```

To see a list of the various options:
```
./safety-gear-python.py -h
```
A user can specify what target device to run on by using the device command-line argument `-d` followed by one of the values `CPU`, `GPU`, `MYRIAD`, `HDDL` or `FPGA `.

### Run on the CPU
Although the application runs on the CPU by default, this can also be explicitly specified through the `-d CPU` command-line argument:
```
./safety-gear-python.py -d CPU -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -e /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -c resources/conf.txt
```

### Run on the Integrated GPU
To run on the integrated Intel GPU with floating point precision 32 (FP32), use the `-d GPU` command-line argument:
```
./safety-gear-python.py -d GPU -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -c resources/conf.txt
```
To run on the integrated Intel® GPU with floating point precision 16 (FP16):
```
./safety-gear-python.py -d GPU -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -c resources/conf.txt
```
### Run on the Intel® Neural Compute Stick
To run on the Intel® Neural Compute Stick, use the `-d MYRIAD` command-line argument:
```
./safety-gear-python.py -d MYRIAD -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -c resources/conf.txt
```
**Note:** The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the `-m <path_to_model>` command-line argument, must be of data type FP16.   

### Run on the HDDL
To run on the HDDL, use the `-d HETERO:HDDL,CPU` command-line argument:
```
./safety-gear-python.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -d HETERO:HDDL,CPU -e /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -c resources/conf.txt
```
**Note:** The HDDL can only run FP16 models. The model that is passed to the application through the `-m <path_to_model>` command-line argument must be of data type FP16.

### Run on the FPGA

Before running the application on the FPGA,  program the AOCX (bitstream) file.
Use the setup_env.sh script from [fpga_support_files.tgz](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz) to set the environment variables.<br>
For example:

```
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```

The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams` folder.<br><br>To program the bitstream use the below command:<br>
```
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_RMNet.aocx
```

For more information on programming the bitstreams, please refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11

To run the application on the FPGA with floating point precision 16 (FP16) use the `-d HETERO:FPGA,CPU` command-line argument:
```
./safety-gear-python.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -d HETERO:FPGA,CPU -e /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -c resources/conf.txt
```

