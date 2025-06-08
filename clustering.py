from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg'
import matplotlib.pyplot as plt


# Initialize Spark session
# Step 1: Create Spark Session

spark = SparkSession.builder \
    .appName("KMeansClustering'") \
    .getOrCreate()
# Load dataset
df = spark.read.csv('iris_dataset.csv', header=True, inferSchema=True)

# Show basic info
print(f"Number of rows: {df.count()}, Number of columns: {len(df.columns)}")
df.printSchema()

# Select features for clustering
input_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Assemble features into a single vector column
vec_assembler = VectorAssembler(inputCols=input_cols, outputCol='features')
df = vec_assembler.transform(df)

# Scale features
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=False)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Determine optimal k using Silhouette score
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(k=k, featuresCol='scaled_features', predictionCol='prediction')
    model = kmeans.fit(df)
    predictions = model.transform(df)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette)
    print(f"Silhouette score for k={k}: {silhouette}")

# Plot Silhouette scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Train KMeans model with k=3
kmeans = KMeans(k=3, featuresCol='scaled_features', predictionCol='prediction')
model = kmeans.fit(df)
predictions = model.transform(df)

# Show cluster sizes
predictions.groupBy('prediction').count().show()

# Convert to Pandas DataFrame for visualization
pandas_df = predictions.select('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'prediction').toPandas()

# 3D Scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pandas_df['sepal_length'], pandas_df['sepal_width'], pandas_df['petal_length'],
           c=pandas_df['prediction'], cmap='viridis', s=50)
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
plt.title('3D Scatter Plot of Clusters')
plt.show()

# Stop Spark session
spark.stop()
