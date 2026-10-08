"""Private Neon S3 objects with short-lived signed playback URLs."""

from urllib.parse import urlsplit

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import RagError


class NeonStorage:
    def __init__(self, settings):
        endpoint = urlsplit(settings.s3_endpoint)
        if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or
                endpoint.password or endpoint.query or endpoint.fragment):
            raise RagError("AWS_ENDPOINT_URL_S3 must be an HTTPS endpoint without credentials or query parameters.")
        if not settings.s3_access_key or not settings.s3_secret_key:
            raise RagError("Configure the Neon object storage credentials before starting the library.")
        self.bucket = settings.bucket
        self.client = boto3.client(
            "s3", endpoint_url=settings.s3_endpoint, region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key, aws_secret_access_key=settings.s3_secret_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                          connect_timeout=5, read_timeout=120,
                          retries={"mode": "standard", "total_max_attempts": 3},
                          request_checksum_calculation="when_required",
                          response_checksum_validation="when_required"),
        )
        self.transfer = TransferConfig(max_concurrency=2)

    def initialize(self):
        # Access policies are provisioned by neon.ts, never by the S3 adapter.
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            raise RagError("The Neon bucket is unavailable. Check its provisioning and storage credentials.") from exc

    def upload(self, key, path, mime_type):
        self.client.upload_file(str(path), self.bucket, key,
                                ExtraArgs={"ContentType": mime_type}, Config=self.transfer)

    def download(self, key, path):
        self.client.download_file(self.bucket, key, str(path), Config=self.transfer)

    def exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def remove(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def playback_url(self, key):
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=900,
        )
