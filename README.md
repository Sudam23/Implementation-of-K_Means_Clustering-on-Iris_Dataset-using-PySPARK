# Distributed KMeans Clustering on Iris Dataset using PySpark

This project demonstrates how to apply the KMeans clustering algorithm to the Iris dataset in a **distributed computing environment** using **Apache Spark** (PySpark). It includes full setup instructions for deploying a Spark standalone cluster and running the ML pipeline with optimal cluster count selection using the **Silhouette Score**.

---

## 📊 Project Features

- Uses **PySpark MLlib** for scalable machine learning.
- Automatically selects the optimal number of clusters using Silhouette Score.
- 3D visualization of clusters with Matplotlib.
- Fully distributed execution on a standalone Spark cluster (multi-node supported).
- Tested with Spark 3.3.2.

---

## 🧰 Technologies Used

- Python 3.10
- PySpark 3.3.2
- Apache Spark Standalone Cluster
- Matplotlib
- Pandas

---

## 🗂 Files

- `clustering.py` – PySpark script for clustering.
- `iris_dataset.csv` – Input dataset.
- `Spark_Cluster setup.txt` – Step-by-step guide to configure a Spark standalone cluster.

---

## ⚙️ Cluster Setup

> Detailed instructions can be found in `Spark_Cluster setup.txt`. Below is a summary.

### 1. Requirements

- Python 3.10
- Java 8 or later (tested with Java 21)
- PySpark (`pip install pyspark==3.3.2`)
- Matplotlib, Pandas

### 2. Single/Multi-Node Cluster

#### Master and Worker Node Setup: Do these following steps in both nodes

```bash
wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
tar -xzf spark-3.3.2-bin-hadoop3.tgz
sudo mv spark-3.3.2-bin-hadoop3 /opt/spark

