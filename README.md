
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
- `Java_Configuration.txt`  – Step to configure Java in Linux Env.
- `DML_Spark.pdf` - Final academic project report.
- `Readme.md` - Project Documentation.
---

## ⚙️ Cluster Setup

> Detailed instructions can be found in `Spark_Cluster setup.txt`. Below is a summary.

### 1. Requirements

- Python 3.10
- Java 8 or later (tested with Java 21)
- PySpark (`pip install pyspark==3.3.2`)
- Matplotlib, Pandas

### 2. Single/Multi-Node Cluster

#### Master and Worker Node Setup: Do these below following steps in both nodes to create Spark Cluster

```bash
1. Create Virtual Env in your desired directory.
2. Activate Env and do pip install --upgrade pip in the terminal.
3. Then in the terminal do: pip install pyspark==3.3.2 (For Standalone Cluster)
4. For multiple node Cluster, download and install Spark:
      i. wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
     ii. tar -xzf spark-3.3.2-bin-hadoop3.tgz
         sudo mv spark-3.3.2-bin-hadoop3 /opt/spark
    iii. echo '
           export SPARK_HOME=/opt/spark
           export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
           export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
           ' >> ~/.bashrc

           source ~/.bashrc


5. On only Master node do that to configure Master:
     i.  cp /opt/spark/conf/spark-env.sh.template /opt/spark/conf/spark-env.sh
         nano /opt/spark/conf/spark-env.sh

    ii.  Add the below two lines in Nano file:    export SPARK_MASTER_HOST= <master-ip address> (e.g, 172.20.251.25)
                                                  export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

   iii.  In Terminal run that to configure Workers:  cp /opt/spark/conf/workers.template /opt/spark/conf/workers
                                                     nano /opt/spark/conf/workers
          
         then add ip list of worker nodes on this nano file(if there is localhost, replace by master node ip and add all the worker node ip addresses)
         (e.g., 172.20.251.22
                172.20.251.25 )          
6. On master node run this to start Master Node:   $SPARK_HOME/sbin/start-master.sh  or $SPARK_HOME/sbin/start-master.sh --host 0.0.0.0
   for stop replace 'start' by 'stop'

7.  Choose another computer for Worker node and find ip address.
8.  Follow 1, 2, 3 and 4.
9.  Start worker from each node:  $SPARK_HOME/sbin/start-worker.sh spark://<MASTER_IP>:7077
10.  Check the website to check the status of Worker node or Cluster.
11. Keep the dataset on the Env Directory (Same_Location) on both Worker and Master Node. (Because there is no common File System e,g., HDFS)
12. Finally Run the ML Model:   $SPARK_HOME/bin/spark-submit  --master spark://172.20.252.53:7077  clustering.py (Replace ip address by Master Node ip address.)

-----
````
### 3. Visit `http://<MASTER_IP>:8080` in your browser to check the Spark UI and check the worker status on the Spark master UI.

## 🚀 How to Run the Project
- Place clustering.py and iris_dataset.csv in the same directory on all nodes (Master and Workers).
- From the Master Node, execute the job using below code:
   `$SPARK_HOME/bin/spark-submit --master spark://<MASTER_IP>:7077 clustering.py`

## 📈 Output
- Prints Silhouette scores for cluster counts from 2 to 9.
- Automatically determines optimal k.
- Displays 3D scatter plot showing clustered data.
- Shows cluster size breakdown in terminal.
  
## 🧠 Learning Outcomes
- Understanding Spark cluster setup manually.
- Distributed execution of MLlib clustering algorithms.
- Evaluating clustering performance using Silhouette Score.
- Visualizing high-dimensional data in 3D space.

## 📌 Notes
- The dataset must be present locally on every node (no shared HDFS).
- Ensure IP configuration and environment paths are consistent across all machines.

## 🖥️ Sample Visualization
> Added screenshots of the silhouette score plot or 3D scatter plot here.

## 🙏 Acknowledgements

Special thanks to:

**Champak Kumar Dutta**  
Assistant Professor, Department of Data Science  
RKMVERI, Belur Math, West Bengal  

For his guidance, mentorship, and continuous encouragement.

## 📜 License

This project is licensed under the **Academic Use Only License**.

- ✅ You are free to use, modify, and distribute this work **for academic, research, and educational purposes only**.
- ❌ Commercial use, redistribution, or integration into proprietary products is **strictly prohibited** without prior written permission.

If you wish to use this project beyond academic contexts, please contact the authors for licensing terms.

## 👨‍💻 Author
- **Sudam Kumar Paul**  
MSc Big Data Analytics | RKMVERI 
- **Kanan Pandit**  
MSc Big Data Analytics | RKMVERI

## 📬 Contact
For queries or improvements, feel free to open an issue or pull request.

**Kanan Pandit**  
🌐 [Portfolio](https://kananpanditportfolio.netlify.app/)  
✉️ kananpandit02@gmail.com  

**Sudam Paul**  
🌐 [Portfolio](https://sudam23.github.io/My_Portfolio/)  
✉️ 2002sudam@gmail.com  

**Institution**  
Ramakrishna Mission Vivekananda Educational and Research Institute  
📍 Belur Math, Howrah, West Bengal   
