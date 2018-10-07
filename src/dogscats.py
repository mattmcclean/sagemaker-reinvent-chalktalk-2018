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
import ast
import argparse
import logging
import json
import os
import io

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from torchvision import transforms

from fastai import *
from fastai.vision import *
from fastai.docs import *

# setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# get the image size from an environment variable for inference
IMG_SIZE = int(os.environ.get('IMAGE_SIZE', '224'))

# define the classification classes
classes = ('cats', 'dogs')

# define the architecture of the app
arch = tvm.resnet34

# By default split models between first and second layer
def _default_split(m:Model): return (m[1],)
# Split a resnet style model
def _resnet_split(m:Model): return (m[0][6],m[1])

_default_meta = {'cut':-1, 'split':_default_split}
_resnet_meta  = {'cut':-2, 'split':_resnet_split }

model_meta = {
    tvm.resnet18 :{**_resnet_meta}, tvm.resnet34: {**_resnet_meta},
    tvm.resnet50 :{**_resnet_meta}, tvm.resnet101:{**_resnet_meta},
    tvm.resnet152:{**_resnet_meta}}

# define the image preprocess steps for inference
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(IMG_SIZE),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Preprocess the image
def preprocess_image(img):
    logger.info("Preprocessing image")
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    return img_tensor

# The train method
def _train(args):
    print(f'Called _train method with batch size: {args.batch_size}, image size: {args.image_size}, epochs: {args.epochs}, learn rate: {args.lr}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device Type: {}".format(device))
    
    print(f'Getting training data from dir: {args.data_dir}')
    data = image_data_from_folder(args.data_dir, ds_tfms=get_transforms(), tfms=imagenet_norm, size=args.image_size)
    print("Creating pretrained conv net")
    learn = ConvLearner(data, tvm.resnet34, metrics=accuracy)
    print("File one cycle")
    learn.fit_one_cycle(1)
    print("Unfreeze and run more cycles")
    learn.fit_one_cycle(6, slice(1e-5,3e-4), pct_start=0.05)
    print('Save model')
    return _save_model(learn.model, args.model_dir)

# save the model
def _save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)
    print('Saved model')

# create the model similar to source code here: https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
def _create_model(arch:Callable):
    print("Creating new model")
    meta = model_meta.get(arch, _default_meta)
    torch.backends.cudnn.benchmark = True
    body = create_body(arch(False), meta['cut'])
    nf = num_features(body) * 2
    head = create_head(nf, len(classes))
    model = nn.Sequential(body, head)
    print("Model created")
    return model
    
# Return the Convolutional Neural Network model
def model_fn(model_dir):
    logger.debug('model_fn')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device Type: {}".format(device))    
    model = _create_model(tvm.resnet34)
    print("Loading model weights")
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))
    print("Model weights loaded")
    model.to(device)
    model.eval()
    return model

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JPEG_CONTENT_TYPE:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device Type: {}".format(device))            
        logger.info('Processing jpeg image.')
        img_pil = PIL.Image.open(io.BytesIO(request_body)).convert('RGB')
        img_tensor = preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor.to(device))
        logger.info("Returning image as PyTorch Variable.")
        return img_variable
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    log_preds = model(input_object).data.numpy()
    print("Raw output")
    print(log_preds)
    print("Exp raw output")
    print(np.exp(log_preds))
    logger.info("Getting best prediction")
    preds = np.argmax(np.exp(log_preds), axis=1)
    
    logger.info("Getting class and confidence score")
    response = {}
    response['class'] = classes[preds.item()]
    response['confidence'] = np.exp(log_preds[:,preds.item()]).item()
    logger.info(response)
    return response

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist-backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    # fast.ai specific parameters
    parser.add_argument('--image-size', type=int, default=224, metavar='IS',
                        help='image size (default: 224)')
    
    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())
