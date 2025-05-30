import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
class S3Sync:
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Syncs a local folder to an AWS S3 bucket.
        
        Args:
            folder (str): Local folder path to sync.
            aws_bucket_url (str): S3 bucket URL where the folder will be synced.
        """
        try:
            # Ensure the folder exists
            if not os.path.exists(folder):
                raise FileNotFoundError(f"The folder {folder} does not exist.")
            
            # Use AWS CLI command to sync the folder to S3
            os.system(f"aws s3 sync {folder} {aws_bucket_url}")
            print(f"Successfully synced {folder} to {aws_bucket_url}")
        except Exception as e:
            raise NetworkSecurityException(sys,e) from e
        
    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        Syncs an AWS S3 bucket to a local folder.
        
        Args:
            folder (str): S3 bucket URL to sync from.
            aws_bucket_url (str): Local directory where the S3 bucket will be synced.
        """
        try:
            # Ensure the local directory exists
            os.makedirs(folder, exist_ok=True)
            
            # Use AWS CLI command to sync the S3 bucket to the local directory
            os.system(f"aws s3 sync {aws_bucket_url} {folder}")
            print(f"Successfully synced {aws_bucket_url} to {folder}")
        except Exception as e:
            raise NetworkSecurityException(sys,e) from e