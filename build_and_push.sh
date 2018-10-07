#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image="sagemaker-fastai-dogscats"

# input parameters
FASTAI_VERSION=${1:-1.0}
PY_VERSION=${2:-py37}

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${FASTAI_VERSION}-gpu-${PY_VERSION}"
docker build -t ${image}:${FASTAI_VERSION}-gpu-${PY_VERSION} --build-arg ARCH=gpu .
docker tag ${image}:${FASTAI_VERSION}-gpu-${PY_VERSION} ${fullname}
docker push ${fullname}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:${FASTAI_VERSION}-cpu-${PY_VERSION}"
docker build -t ${image}:${FASTAI_VERSION}-cpu-${PY_VERSION} --build-arg ARCH=cpu .
docker tag ${image}:${FASTAI_VERSION}-cpu-${PY_VERSION} ${fullname}
docker push ${fullname}
