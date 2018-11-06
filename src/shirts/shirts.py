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
import glob
from io import BytesIO

import torch
import torch.nn.functional as F

import fastai
from fastai import *
from fastai.vision import *

# setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# get the image size from an environment variable for inference
IMG_SIZE = int(os.environ.get('IMAGE_SIZE', '224'))

# define the classification classes
CLASSES = ['metal', 'sport']

# The train method
def _train(args):
    print(f'Called _train method with model arch: {args.model_arch}, batch size: {args.batch_size}, image size: {args.image_size}, epochs: {args.epochs}')
    print(f'Getting training data from dir: {args.data_dir}')
    data = ImageDataBunch.from_folder(args.data_dir, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=args.image_size, num_workers=4, bs=args.batch_size).normalize(imagenet_stats)
    print(f'Model architecture is {args.model_arch}')
    arch = getattr(models, args.model_arch)
    print("Creating pretrained conv net")
    learn = create_cnn(data, arch, metrics=accuracy)
    print("Fit four cycles")
    learn.fit_one_cycle(4)
    print(f'Unfreeze and run {args.epochs} more cycles')
    learn.unfreeze()
    learn.fit_one_cycle(args.epochs, max_lr=slice(3e-5,3e-4))
    path = Path(args.model_dir)
    print(f'Saving model weights to dir: {args.model_dir}')
    learn.save(path/args.model_arch)

# Return the Convolutional Neural Network model
def model_fn(model_dir):
    logger.debug('model_fn')
    fastai.defaults.device = torch.device('cpu')
    # get the model architecture from name of saved model weights
    arch_name = os.path.splitext(os.path.split(glob.glob(f'{model_dir}/resnet*.pth')[0])[1])[0]
    print(f'Model architecture is: {arch_name}')
    arch = getattr(models, arch_name)
    empty_data = ImageDataBunch.single_from_classes(Path('/tmp'), CLASSES, 
        tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(empty_data, arch, pretrained=False)
    learn.load(Path(model_dir)/arch_name)
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    if content_type == JPEG_CONTENT_TYPE:
        img = open_image(BytesIO(request_body))
        return img
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    predict_class,_,predict_values = model.predict(input_object)
    print(f'Predicted class is {predict_class}')
    print(f'Predicted values are {predict_values}')
    preds = F.softmax(predict_values, dim=0)
    conf_score, indx = torch.max(preds, dim=0)
    print(f'Softmax confidence score is {conf_score.item()}')
    response = {}
    response['class'] = predict_class
    response['confidence'] = conf_score.item()
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
    parser.add_argument('--model-arch', type=str, default='resnet34', metavar='MA',
                        help='model arch (default: resnet34)')
    
    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())
