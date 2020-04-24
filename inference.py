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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.input_shape = None
        
    def load_model(self):
        ### TODO: Load the model ###
        plugin = IECore()
        model_xml = "/home/workspace/model/MobileNetSSD_deploy10695.xml"
        model_bin = "/home/workspace/model/MobileNetSSD_deploy10695.bin"
        net = IENetwork(model=model_xml, weights=model_bin)
        plugin.add_extension(CPU_EXTENSION, "CPU")#must add
        
        ### TODO: Check for supported layers ###
        supported_layers = plugin.query_network(network=net, device_name="CPU")
        
        unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        
        self.exec_network = plugin.load_network(net, "CPU")
        print("IR successfully loaded into Inference Engine.")
        self.plugin = plugin
        self.network = net
        
        self.input_blob = next(iter(net.inputs))#added
        self.output_blob = next(iter(self.network.outputs))
        self.input_shape = net.inputs[self.input_blob].shape
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.input_shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        exec_network = self.exec_network
        input_blob = self.input_blob
        exec_network.start_async(request_id = 0, inputs = {input_blob : image})
        
        self.exec_network = exec_network
        
        
        
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return exec_network

    def wait(self, exec_network):
        status = None
        while True:
            status = exec_network.requests[0].wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
            
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[0].outputs[self.output_blob]
