import argparse
import cv2
from inference import Network
import numpy as np
import time
import socket
import json
import logging as log
import paho.mqtt.client as mqtt
import sys #for ffmpeg server


from argparse import ArgumentParser

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


INPUT_STREAM = "resources/Pedestrian_Detect_2_1_1.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
OB_MODEL = "/home/workspace/model/MobileNetSSD_deploy10695.xml"

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       
    pt_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=False, default = OB_MODEL)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-pt", help=pt_desc, default=0.3)
    optional.add_argument("-l", help = "extention", default = CPU_EXTENSION )
    args = parser.parse_args()

    return args

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
            publish_frame_duration = stat['frame_duration']/10
#             publish_frame_duration = publish_frame_duration / 10 # 10 is frame rate of video
            client.publish("person/duration",json.dumps({"duration": publish_frame_duration}))
            stat['frame_duration'] = 0
            stat['frame_buffer'] = 0

        else:
            stat['frame_buffer'] += 1

    if stat['is_person_present'] is True and person_detected is True :
        stat['frame_duration'] = stat['frame_duration'] + 1 + stat['frame_buffer']
        stat['frame_buffer']  = 0  
   
    return stat, people_count




def draw_boxes(frame, result, args, width, height, prob_threshold, person_detected):
    '''
    Draw bounding boxes onto the frame.
    '''
    person_detected = False
    for box in result[0][0]: # Output shape is 1x1x100x7

        if int(box[1]) == 1 :
            conf = box[2]
            if conf >= 0.3:
                person_detected = True
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255,1))

    return frame, person_detected


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args,client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    frame_no = 0
    stat = {'is_person_present' : False,
            'begin_frame' : 0,
            'end_frame' : 0,
            'frame_duration' : 0,
            'frame_buffer' : 0}
    person_detected = False
    people_count = 0
    frame_thresh = 10
    prev_total_count = 0
    curr_total_count = 0 
    last_count = 0
    current_count = 0

    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.pt  #args.prob_threshold hardcoded for testing

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args)
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Create a video writer for the output video
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    k = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        
        # Read the next frame
        flag, frame = cap.read()
        
        
        
        if not flag:
            break
            

        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###

        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)        
        
        ### TODO: Start asynchronous inference for specified request ###
  
        tmp_net = infer_network.exec_net(p_frame)# tmp_net = exec_net 
        
        ### TODO: Wait for the result ###
        if infer_network.wait(tmp_net) == 0:
            result = infer_network.get_output()              
            resframe,person_detected = draw_boxes(frame, result, args, width, height,prob_threshold,person_detected)
            out.write(resframe)
            frame_no +=1
            
            stat, people_count = get_stat(stat, frame_no, people_count, frame_thresh, person_detected,client)
            
            last_count = current_count
            if stat['is_person_present'] == True :
                current_count = 1
            else:
                current_count = 0
            total_count = people_count
            prev_total_count = curr_total_count
            curr_total_count = total_count
#             duration = int(stat['frame_duration']/24)
            

            
            # Person duration in the video is calculated
#             if current_count < last_count:
#                 duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
#                 client.publish("person/duration",
#                                json.dumps({"duration": duration}))
             
            client.publish("person", json.dumps({"count": current_count}))
            
   
            
            #break if escape key is pressed
            ### current_count, total_count and duration to the MQTT server ###

#             print('current count:{}  total_count:{}  duration:{}'.format(current_count, total_count, duration))
#             client.publish("person", json.dumps({"count": current_count, "total": total_count}))
#             client.publish("person/duration", json.dumps({"duration": duration}))
            
            ### TODO: Send the frame to the FFMPEG server ###
            # When new person enters the video
            if current_count > last_count:
#                 start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
            
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
            
            if key_pressed == 27:
                break
    out.release()
    client.disconnect()
    cap.release()
#     print('stats is {} \n person counted = {}'.format(stat, people_count))
    return


def main():
    args = get_args() # FIXME add Build_parser
#     args = build_argparser().parse_args()
    client = connect_mqtt()
    infer_on_stream(args,client)
#     infer_on_stream(args)


if __name__ == "__main__":
    main()
