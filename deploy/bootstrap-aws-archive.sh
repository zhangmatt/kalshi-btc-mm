#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:?usage: bootstrap-aws-archive.sh INSTANCE_ID [BUCKET]}"
REGION="${AWS_REGION:-us-east-1}"
BUCKET="${2:-kalshi-btc-mm-${INSTANCE_ID}-${REGION}}"
ROLE="kalshi-btc-mm-archive"
PROFILE="kalshi-btc-mm-archive"

if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
fi

aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":false}]}'
aws s3api put-bucket-versioning --bucket "$BUCKET" --versioning-configuration Status=Enabled

TLS_POLICY=$(printf '%s' '{"Version":"2012-10-17","Statement":[{"Sid":"DenyInsecureTransport","Effect":"Deny","Principal":"*","Action":"s3:*","Resource":["arn:aws:s3:::'"$BUCKET"'","arn:aws:s3:::'"$BUCKET"'/*"],"Condition":{"Bool":{"aws:SecureTransport":"false"}}}]}')
aws s3api put-bucket-policy --bucket "$BUCKET" --policy "$TLS_POLICY"

TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" >/dev/null
fi

POLICY=$(printf '%s' '{"Version":"2012-10-17","Statement":[{"Sid":"BucketMetadata","Effect":"Allow","Action":["s3:GetBucketLocation","s3:ListBucket"],"Resource":"arn:aws:s3:::'"$BUCKET"'"},{"Sid":"VerifiedArchiveObjects","Effect":"Allow","Action":["s3:PutObject","s3:GetObject"],"Resource":"arn:aws:s3:::'"$BUCKET"'/kalshi-btc15m/*"}]}')
aws iam put-role-policy --role-name "$ROLE" --policy-name kalshi-btc-mm-archive --policy-document "$POLICY"

if ! aws iam get-instance-profile --instance-profile-name "$PROFILE" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PROFILE" >/dev/null
fi

PROFILE_ROLE=$(aws iam get-instance-profile --instance-profile-name "$PROFILE" \
  --query 'InstanceProfile.Roles[0].RoleName' --output text)
if [[ -z "$PROFILE_ROLE" || "$PROFILE_ROLE" == "None" ]]; then
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE" --role-name "$ROLE"
  sleep 10
elif [[ "$PROFILE_ROLE" != "$ROLE" ]]; then
  echo "instance profile $PROFILE already contains unexpected role $PROFILE_ROLE" >&2
  exit 1
fi

ASSOCIATION=$(aws ec2 describe-iam-instance-profile-associations --region "$REGION" \
  --filters "Name=instance-id,Values=$INSTANCE_ID" \
  --query 'IamInstanceProfileAssociations[0].AssociationId' --output text)
if [[ -z "$ASSOCIATION" || "$ASSOCIATION" == "None" ]]; then
  aws ec2 associate-iam-instance-profile --region "$REGION" --instance-id "$INSTANCE_ID" \
    --iam-instance-profile "Name=$PROFILE" >/dev/null
else
  aws ec2 replace-iam-instance-profile-association --region "$REGION" \
    --association-id "$ASSOCIATION" --iam-instance-profile "Name=$PROFILE" >/dev/null
fi

echo "KALSHI_S3_BUCKET=$BUCKET"
echo "Attached instance profile $PROFILE to $INSTANCE_ID in $REGION"
