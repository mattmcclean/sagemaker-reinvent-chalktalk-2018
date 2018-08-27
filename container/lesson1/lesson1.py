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

import os

import torch

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

classes = ('cats', 'dogs')

arch = resnet34

def _train(args):
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        logger.info(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Device Type: {}".format(device))

    logger.debug("Loading data transformer")
    tfms = tfms_from_model(arch, args.image_size, aug_tfms=transforms_side_on, max_zoom=1.1)
    
    logger.debug("Creating image classifier")
    data = ImageClassifierData.from_paths(args.data_dir, bs=args.batch_size, tfms=tfms)
    
    logger.info("Creating pretrained conv net")
    learn = ConvLearner.pretrained(arch, data, precompute=True)
    
    logger.debug("Starting training...")
    learn.fit(args.lr, 1)
    learn.precompute=False
    learn.fit(args.lr, args.epochs, cycle_len=1)
    logger.info("Training finished")
    print('Finished Training')
    return _save_model(learn.model, args.model_dir)


def _save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)


def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvLearner(arch, len(classes), False, False).model
    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
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