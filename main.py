"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def get_stat(stat, frame_no, people_count, frame_thresh, person_detected,client):
    
    if stat['is_person_present'] is False and person_detected is True :
        stat['is_person_present'] = True
        stat['begin_frame'] = frame_no
        stat['end_frame'] = 0
        stat['frame_duration'] = 1
        people_count += 1

    if stat['is_person_present'] is True and person_detected is False:
        diff =frame_no - (stat['begin_frame'] + stat['frame_duration'])
        if diff >= frame_thresh:
            stat['is_person_present'] = False
            stat['end_frame'] = frame_no
            publish_frame_duration = stat['frame_duration']/10 #get duration by dividing by frame rate
            client.publish("person/duration",json.dumps({"duration": publish_frame_duration}))
            stat['frame_duration'] = 0
            stat['frame_buffer'] = 0

        else:
            stat['frame_buffer'] += 1

    if stat['is_person_present'] is True and person_detected is True :
        stat['frame_duration'] = stat['frame_duration'] + 1 + stat['frame_buffer']
        stat['frame_buffer']  = 0  
   
    return stat, people_count


def draw_boxes(frame, result, args, width, height,prob_threshold, person_detected):#draw boxes
    '''
    draw bounding boxes onto the frame
    '''
    for box in result[0][0]: #output shape is 1x1x100x7
        if int(box[1]) == 1: # class is human
            conf = box[2] #confidence score
            if conf >= 0.3:
                person_detected = True
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.recangle(frame, (xmin, ymin), (xmax, ymax),(0,0,255), 1)
    return frame, person_detected


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    """
    variables to use
    """
    frame_no = 0 #keep track of current frame number
    
    #stores the calculated stats
    stat = {'is_person_present' : False, 
            'begin_frame' : 0,
            'end_frame' : 0,
            'frame_duration' : 0,
            'frame_buffer' : 0}
    person_detected = False #true if person got detected in current frame
    people_count = 0 #total number of people counted
    frame_thresh = 25 #to avoid error in people count
    prev_total_count = 0 # total count in previous frame
    curr_total_count = 0 # total count in current frame
    last_count = 0 # no. of people counted in previous frame
    current_count = 0 # no.of people counted in current frame
    
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(args.model,args)
    net_input_shape = infer_network.get_input_shape()
    # Handle the input stream 
    cap = cv2.VideoCapture(args.input)
    # Get and open video capture
    
    cap.open(args.input)
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Create a video writer for the output video
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Loop until stream is over 
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Pre-process the image as needed 

        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)        
        
        # Start asynchronous inference for specified request 
  
        tmp_net = infer_network.exec_net(p_frame)# tmp_net = exec_net 

        # Wait for the result
        if infer_network.wait(tmp_net) == 0:
            #  Get the results of the inference request
            result = infer_network.get_output()
            resframe, person_detected = draw_boxes(frame, result, args, width, height, prob_threshold,person_detected)
            
           # write output frame
            out.write(resframe)
           # Extract desired stats from the results
            stat, people_count = get_stat(stat, frame_no, people_count, frame_thresh, person_detected,client)
            
            last_count = current_count
            if stat['is_person_present'] == True :
                current_count = 1
            else:
                current_count = 0
            total_count = people_count
            prev_total_count = curr_total_count
            curr_total_count = total_count
            
            ### send information to the MQTT server ###
            # When new person enters the video
            if current_count > last_count:
                client.publish("person", json.dumps({"total": total_count}))
                
            client.publish("person", json.dumps({"count": current_count}))
            
            # break if escape key is pressed
            # Send the frame to the FFMPEG server
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
            
            if key_pressed == 27:
                break

        ### TODO: Write an output image if `single_image_mode` ### #fixme
        out.release()
        client.disconnect()
        cap.release()
        return


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
