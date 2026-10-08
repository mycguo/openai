"""Private MinIO objects; only short-lived signed playback URLs leave the server."""

from datetime import timedelta

from minio import Minio
from minio.error import S3Error
import urllib3


class Storage:
    def __init__(self, settings):
        self.bucket = settings.bucket
        options = dict(access_key=settings.minio_access_key, secret_key=settings.minio_secret_key,
                       secure=settings.minio_secure, region="us-east-1")
        self.client = Minio(settings.minio_endpoint, **options, http_client=urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=5, read=120), retries=False))
        self.public = Minio(settings.minio_public_endpoint, **options)

    def initialize(self):
        if not self.client.bucket_exists(self.bucket):
            try:
                self.client.make_bucket(self.bucket)
            except S3Error as exc:
                if exc.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                    raise

    def upload(self, key, path, mime_type):
        self.client.fput_object(self.bucket, key, str(path), content_type=mime_type)

    def download(self, key, path):
        self.client.fget_object(self.bucket, key, str(path))

    def exists(self, key):
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise

    def remove(self, key):
        self.client.remove_object(self.bucket, key)

    def playback_url(self, key):
        return self.public.presigned_get_object(self.bucket, key, expires=timedelta(minutes=15))
