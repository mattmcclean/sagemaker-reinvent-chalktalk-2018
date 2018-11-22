#!/bin/bash
  
# input parameters
IAM_USERNAME=${1:-sagemakernb}

IAM_USER_DETAILS=$(aws iam get-user --user-name ${IAM_USERNAME} 2> /dev/null)

if [[ $? == 0 ]]; then
    echo "IAM user: \"${IAM_USERNAME}\" already exists"
    exit 1
else
    # create the IAM user
    echo "Creating IAM user"
    aws iam create-user --user-name ${IAM_USERNAME} > /dev/null 2>&1

    # create the access keys for programmatic access
    echo "Creating access keys"
    ACCESS_KEYS=$(aws iam create-access-key --user-name ${IAM_USERNAME})
    accesskey=$(echo $ACCESS_KEYS | jq '.AccessKey.AccessKeyId' -r)
    secretkey=$(echo $ACCESS_KEYS | jq '.AccessKey.SecretAccessKey' -r)

    # attach the SageMaker policy to the user
    echo "Attaching SageMaker policy to user"
    aws iam attach-user-policy --user-name ${IAM_USERNAME} --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"

    # save the credentials to AWS Secrets Manager
    echo "Saving credentials in AWS Secrets Store"
    aws secretsmanager create-secret --name SageMakerNbAccessKey --secret-string "${accesskey}"
    aws secretsmanager create-secret --name SageMakerNbSecretKey --secret-string "${secretkey}"
    echo "Successfully created IAM credentials and saved to AWS Secret store with names: \"SageMakerNbAccessKey\" and \"SageMakerNbSecretKey\""
fi