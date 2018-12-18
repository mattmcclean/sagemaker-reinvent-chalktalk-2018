# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import logging
import json
import os
import glob
import io
import sys
import time

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms

# setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# get the image size from an environment variable for inference
IMG_SIZE = int(os.environ.get('IMAGE_SIZE', '224'))

classes = []

# Image pre-processing transforms
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# processing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# Return the Convolutional Neural Network model
def model_fn(model_dir):
    global classes
    logger.debug(' v')
    print(f'Model dir is {model_dir}')
    # get the classes from saved 'classes.txt' file
    with open(f'{model_dir}/classes.txt', 'r') as f:
        classes = f.read().splitlines()
    print(f'Classes are {classes}')    
    model_path = glob.glob(f'{model_dir}/*_jit')[0]
    print(f'Model path is {model_path}')
    return torch.jit.load(model_path, map_location=torch.device('cpu'))

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE:
        img = Image.open(io.BytesIO(request_body))
        return _normalize_img(img)   
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    
# Normalise the image using the Torchvision library
def _normalize_img(img):
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_values = model(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    preds = F.softmax(predict_values, dim=1)
    conf_score, indx = torch.max(preds, dim=1)
    predict_class = classes[indx]
    print(f'Predicted class is {predict_class}')
    print(f'Predicted values are {predict_values}')
    print(f'Softmax confidence score is {conf_score.item()}')
    response = {}
    response['class'] = str(predict_class)
    response['confidence'] = conf_score.item()
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    
