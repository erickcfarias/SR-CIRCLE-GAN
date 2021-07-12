import logging
from botocore.exceptions import ClientError
import boto3
import os


class S3Manager:

    def __init__(self, key, secret):

        self.session = boto3.Session(
            aws_access_key_id=key,
            aws_secret_access_key=secret
        )
        self.s3 = self.session.resource(
            's3', region_name="us-east-1"
        )
        self.s3_client = self.session.\
            client('s3')

    def upload_file(self, file_name, bucket, object_name=None):
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then
            file_name is used
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name

        # Upload the file
        try:
            self.s3_client.\
                upload_file(file_name, bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def download_file(self, file_name, bucket, object_name=None):
        """Download a file from a S3 bucket

        :param file_name: file path name to save
        :param bucket: Bucket to download from
        :param object_name: S3 object name
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name

        # Upload the file
        try:
            self.s3_client.download_file(
                bucket, object_name, file_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def clear_folder(self, bucket: str, prefix: str):
        bucket = self.s3.Bucket(bucket)

        if prefix[-1] == '/':
            prefix = prefix[0:-1]

        bucket.objects.filter(
            Prefix=prefix
        ).delete()

    def download_all_files(self, bucket, version_name, save_dir):
        # select bucket
        my_bucket = self.s3.Bucket(bucket)
        # download file into current directory
        for s3_object in my_bucket.objects.all():
            if version_name + '/' in s3_object.key:
                my_bucket.download_file(
                    s3_object.key,
                    os.path.join(
                        save_dir,
                        s3_object.key.replace(version_name + '/', '')
                    )
                )
