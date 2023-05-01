import argparse
import getpass
from hdfs import InsecureClient


# Initialize the argument parser
parser = argparse.ArgumentParser(description='Upload the data folder to HDFS')
parser.add_argument('-lp', '--local_path', help='Path to the local folder to upload', default='./data')
parser.add_argument('-hp', '--hdfs_path', help='Path to the hdfs folder', default='/data')
args = parser.parse_args()

# Get the user name
user_name = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=user_name)

# Create the hdfs directory if it does not exist
if not hdfs_client.status(args.hdfs_path, strict=False):
    hdfs_client.makedirs(args.hdfs_path)

# Upload the folder
hdfs_client.upload(args.hdfs_path, args.local_path)
