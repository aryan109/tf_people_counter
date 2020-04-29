# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

I have used SSD Mobilenet V2 object detection model from the Tensorflow model zoo. It is pretty fast and small sized model which is ideal for doing inference on videos. 

To convert the model into it's intermediate representation i have used following command:-
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

the complete steps to convert model is present in model.ipynb file.

while conversion I have used ssd_v2_support.json file and passed it to --tensorflow_use_custom_operations_config paraameter. 

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...
the size of intermediate representation is: my_model.bin = 65M and my_model.xml = 110K.


The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are :-
1) During the lockdown period ,this app can be used to locate those areas where more number of people are accumulating and voilating lockdown.
2) Counting the number of people in real time in a particular frame to get an estimate of rush in the store.
3) find the average duration of time spent by coustmers in a store.
4) finding the peak rush hours in a store.

Each of these use cases would be useful because it will help in improving the profitability and do survilliance in very effective and efficient manner.



## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:-

for this application to be effective there should be a good amount of lightning present and artificial lightning can be provided during low light conditions.

I have used SSD Mobilenet V2 model which is has very less inference time along with pretty good accuracy. This model can predict big objects like humans in this case pretty easily with above 90% accuracy. This model is very suitable for detection on video streams due to its less inference time.

This app is created such that, at before doing the inference the video frame is resized according to the input shape to model. Standard frame size like (640 x 420) and (1280 x 720)
are recommended for optimal results.


## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
