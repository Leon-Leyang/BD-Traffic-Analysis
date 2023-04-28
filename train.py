import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors


# The files for features and labels
feature_file = "/home/leon/DAP/data/train_set/X_train_Austin.npy"
label_file = "/home/leon/DAP/data/train_set/y_train_Austin.npy"

# Create the spark session
spark = SparkSession.builder.appName("Train").getOrCreate()

# Load ndarrays
features = np.load(feature_file, allow_pickle=True).astype(float)
labels = np.load(label_file, allow_pickle=True).astype(int)

# Convert feature and label arrays to list of tuples
rows = [(Vectors.dense(f), l.tolist()) for f, l in zip(features, labels)]


# Create DataFrame from list of tuples with column names "feature" and "label"
df = spark.createDataFrame(rows, ["feature", "label"])

# Initialize the model
lr = LogisticRegression(featuresCol="feature", labelCol="label")

# Train the model
model = lr.fit(df)

