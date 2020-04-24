import argparse
import cv2
from test_infer import Network
import numpy as np

INPUT_STREAM = "edit_test.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
OB_MODEL = "/home/workspace/model/MobileNetSSD_deploy10695.xml"

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
    ###       2) The user choosing the color of the bounding boxes
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=False, default = OB_MODEL)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.2)
    args = parser.parse_args()

    return args


def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        if int(box[1]) == 1 :
            conf = box[2]
            if conf >= args.ct:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255,1))

    return frame


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args):#argument client removed for testing
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = float(0.3)  #args.prob_threshold hardcoded for testing

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model()
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Create a video writer for the output video
    out = cv2.VideoWriter('out4.mp4', 0x00000021, 30, (width,height))
    k = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        
        # Read the next frame
        flag, frame = cap.read()
#         print('frame {} , flag {}'.format(k,flag))
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
            resframe = draw_boxes(frame, result, args, width, height)          
            out.write(resframe)
            
            #break if escape key is pressed
            if key_pressed == 27:
                break
    out.release()
    return


def main():
    args = get_args()
    infer_on_stream(args)


if __name__ == "__main__":
    main()
