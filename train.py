import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# The files for features and labels
train_feature_file = "/home/leon/DAP/data/train_set/X_train_Austin.npy"
train_label_file = "/home/leon/DAP/data/train_set/y_train_Austin.npy"
test_feature_file = "/home/leon/DAP/data/train_set/X_test_Austin.npy"
test_label_file = "/home/leon/DAP/data/train_set/y_test_Austin.npy"

# Create the spark session
spark = SparkSession.builder.appName("Train").getOrCreate()

# Load ndarrays
train_features = np.load(train_feature_file, allow_pickle=True).astype(float)
train_labels = np.load(train_label_file, allow_pickle=True).astype(int)
test_features = np.load(test_feature_file, allow_pickle=True).astype(float)
test_labels = np.load(test_label_file, allow_pickle=True).astype(int)

# Convert feature and label arrays to list of tuples
train_data = [(Vectors.dense(f), l.tolist()) for f, l in zip(train_features, train_labels)]
test_data = [(Vectors.dense(f), l.tolist()) for f, l in zip(test_features, test_labels)]

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

