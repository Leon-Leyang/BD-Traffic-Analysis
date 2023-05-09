import getpass
import numpy as np
from hdfs import InsecureClient
from io import BytesIO
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def load_hdfs_npy(hdfs_client, path):
	with hdfs_client.read(path) as f:
		content = f.read()
	return np.load(BytesIO(content), allow_pickle=True)	
	

if __name__ == "__main__":
	# Get the user name
	user_name = getpass.getuser()

	# Initialize the hdfs client
	hdfs_client = InsecureClient('http://localhost:9870', user=user_name)

	# Set the name of the city whose data will be used
	city = "Austin"

	# The files for features and labels
	train_x_file = f"/data/train_set/x_train_{city}.npy"
	train_y_file = f"/data/train_set/y_train_{city}.npy"
	test_x_file = f"/data/train_set/x_test_{city}.npy"
	test_y_file = f"/data/train_set/y_test_{city}.npy"

	# Create the spark session
	spark = SparkSession.builder.appName("Train").getOrCreate()

	# Load ndarrays
	train_x = load_hdfs_npy(hdfs_client, train_x_file).astype(float)
	train_y = load_hdfs_npy(hdfs_client, train_y_file).astype(int)
	test_x = load_hdfs_npy(hdfs_client, test_x_file).astype(float)
	test_y = load_hdfs_npy(hdfs_client, test_y_file).astype(int)

	# Convert feature and label arrays to list of tuples
	train_data = [(Vectors.dense(f), l.tolist()) for f, l in zip(train_x, train_y)]
	test_data = [(Vectors.dense(f), l.tolist()) for f, l in zip(test_x, test_y)]

	# Create DataFrame from list of tuples with column names "feature" and "label"
	train_df = spark.createDataFrame(train_data, ["feature", "label"])
	test_df = spark.createDataFrame(test_data, ["feature", "label"])

	# Initialize the model
	lr = LogisticRegression(featuresCol="feature", labelCol="label")

	# Train the model
	model = lr.fit(train_df)

	# Evaluate the model
	preds = model.transform(test_df)
	accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
	recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
	f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
	accuracy = accuracy_evaluator.evaluate(preds)
	recall = recall_evaluator.evaluate(preds)
	f1 = f1_evaluator.evaluate(preds)

	print(f"Accuracy: {accuracy}")
	print(f"Recall: {recall}")
	print(f"F1: {f1}")

	# Stop the session
	spark.stop()

