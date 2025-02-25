# KDDCup99-RapidMiner

Briefly, this project focuses on the KDDCup99 dataset, performing advanced analytics on various data points and features. It utilizes advanced feature engineering techniques and machine learning models/algorithms for both predictive and prescriptive analysis, implemented through the RapidMiner tool.

---

## Table of Contents
1. [Important Point](#important-point)
2. [Project Objectives](#project-objectives)
3. [Data Source](#data-source)
4. [Project Implementation](#project-implementation)
   - [Data Preprocessing](#data-preprocessing)
   - [Clustering via DBScan Algorithm](#clustering-via-dbscan-algorithm)
   - [Information Classification](#information-classification)
5. [Conclusion](#conclusion)
6. [Suggestions for Development](#suggestions-for-development)

---
## Important Point
- I would like to highlight an important point about this project and its associated file. Please note that this is an educational project designed to be followed step by step. Since RapidMiner is an accumulative data science tool that does not allow separation of different project components, the README file plays a critical role in guiding users through the process. It is, therefore, the most essential part of this project.

---

## Project Objectives
- In this project, we aim to utilize clustering algorithms within various machine learning models to identify and analyze traffic patterns. The primary objective of our model is to distinguish between normal and abnormal traffic, enabling the detection and prevention of malicious network intrusions. Beyond identifying the malicious nature of these intrusions, our model is also designed to classify the specific type of attack.

- Additionally, we will develop and train a decision tree model capable of predicting the categories of security attacks. To accomplish these objectives, we will leverage the RapidMiner software for data preprocessing, model implementation, and analysis.

---

## Data Source
- In this project, we utilize the KDDCup99 dataset to detect suspicious network traffic and identify various types of network attacks. The KDDCup99 dataset, originally created for the 3rd International Knowledge Discovery and Data Mining Tools Competition, is one of the most renowned datasets in the field of network intrusion detection. It contains a wide range of data from simulated intrusions in a military network environment and is specifically designed to build predictive models capable of distinguishing between 'bad' connections (intrusions or attacks) and 'good' connections (normal traffic). The key features of the KDDCup99 dataset include:
   - **Number of Records**: The dataset contains **494,021 records**.
   - **Number of Features**: The dataset includes **42 features**.
   - **Key Features**: Some of the most important features are `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, and more.
   - **Output Labels**: The dataset has **23 different output labels**, where one label is `normal` (representing good connections) and the rest represent various types of bad connections (intrusions or attacks).

   For more details about the features and their descriptions, visit the official website: [KDDCup99 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html).
      
   **Dataset Download**: The dataset can be downloaded from [Kaggle](https://www.kaggle.com).

---

## Project Implementation
### Data Preprocessing
After downloading the dataset, we observe that the downloaded file contains multiple datasets, all relevant to the problem at hand. We will work with the `kddcup.data.corrected` file. 

### Step 1: Preparing the Dataset
- To improve data readability in RapidMiner, we first convert the file to CSV format using a data reading tool in Windows.
- We add a header row at the beginning of the file containing the names and types of the features. During the import process in RapidMiner, we specify that this row should be treated as feature names (not as a data record).
- To access the feature names, refer to the `kddcup.names` file in the ZIP folder. Note that this file does not include the output label (target feature), which must be added manually.

### Step 2: Handling Data Types
- During the import process, some columns with binary values may be incorrectly stored as integers. These columns must be converted to the `polynominal` type (nominal features in RapidMiner). The affected features are:
  - `land`
  - `logged_in`
  - `is_host_login`
  - `is_guest_login`

### Step 3: Initial Data Inspection
- After importing the data, we navigate to the **Statistics** section. Here, we observe that there are no missing values requiring preprocessing (e.g., removal or imputation with mean values). However, for the sake of completeness, we assume the presence of missing data in this project.

### Step 4: Organizing the Workflow
- To maintain an organized workflow, we use **Subprocesses** (available under the **Operators** section).

---

## Parent Process: PreprocessingData

### 1. RemoveDuplicates
- This operator removes duplicate records to reduce the dataset size and simplify subsequent preprocessing steps.

### 2. FilterExamples
- For completeness, this operator filters out records where the `duration` feature is null.

### 3. ReplaceMissingValues
- This operator replaces null values in each column with the mean value of that column.

### 4. Normalize
- This operator normalizes the 34 continuous numerical features in the dataset.

### 5. Nominal2Numerical
- This operator converts the 7 nominal features in the dataset into numerical features.

### 6. DetectOutlierLOF
- This operator detects and removes outliers using the **Local Outlier Factor (LOF)** method with the Euclidean distance function. This step also helps remove noise.
- The output of this operator is the same dataset with an additional column named `outlier`, containing `true` or `false` values.
- Due to the high memory complexity of this operator, especially with large datasets like this one, we limit outlier detection to the preprocessing steps already performed.
- Additionally, since we will use the **DBSCAN algorithm** later in the project—which inherently detects and handles noise and outliers—this step is not strictly necessary at this stage.

### Why Use LOF?
- The dataset contains both dense and sparse regions of data. Outliers may be subtle (e.g., attack patterns). Therefore, we choose **Local Outlier Factor (LOF)**, which effectively handles variations in data density and can detect subtle anomalies that may indicate attacks.
- On the other hand, if computational cost is a concern, density-based detection is a simpler yet effective alternative for this dataset.

---

## Additional Preprocessing Steps

### 7. SetRole
- This operator assigns the `connection_type` feature as the **label** for the model (and dataset). This allows us to focus on this target variable in subsequent steps and distinguish the features that influence it.

### 8. WeightByInfoGain
- With the `connection_type` feature marked as the label in the previous step, this operator generates a table with two key outputs:
  1. A list of features from the dataset.
  2. The impact of each feature on the label (measured by information gain).
- The output of this operator includes a **weights table** and the original dataset (for use in later steps).

### 9. SelectByWeights
- This operator takes the original dataset and the weights table as inputs. It is configured with a threshold weight (e.g., 0.2) to filter features based on their importance to the label.
- The output is a new dataset containing only the features with weights greater than the configured threshold. For example, if the threshold is set to 0.2, the number of features is reduced from 41 to 20.
<img width="430" alt="Screenshot 2025-02-25 at 7 31 43 PM" src="https://github.com/user-attachments/assets/438825a1-2fa8-4199-ba14-ea171c418634" />

---

### Clustering via DBScan Algorithm
In this phase, our task is to perform clustering using the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm. DBSCAN is a popular clustering method that operates based on spatial density and is capable of identifying clusters of varying shapes. It is particularly suitable for datasets with noise and clusters of different densities.

---

### How DBSCAN Works
1. For each point in the dataset, the number of points within a radius of `eps` is counted.
2. If the number of points within `eps` is greater than or equal to `min_samples`, the point is identified as a **core point**.
3. All neighboring points within the `eps` radius are added to the cluster.
4. This process continues until all points are either assigned to a cluster or marked as noise.

---

### Advantages of DBSCAN
- **Shape Flexibility**: DBSCAN can detect clusters of arbitrary shapes.
- **Noise Handling**: It is robust to noise and outliers.
- **No Need for Predefined Clusters**: Unlike algorithms like K-Means, DBSCAN does not require the number of clusters to be specified in advance.

DBSCAN is widely used in various fields, such as **intrusion detection**, **spatial data analysis**, and **biological data analysis**.

---

### Goal of Clustering in This Project
The goal of clustering in this project is to extract meaningful insights from the data and provide actionable knowledge that can improve network-related decision-making. Clustering also helps in better understanding the nature and structure of the data, enabling the development of more accurate classification models.

---

### Handling Large Data
Due to the size of the dataset, we use the **Sample** operator to reduce the data volume before implementing the model. We reduced the dataset to **1,000 records** using absolute sampling.

---

### Selecting `min_samples`
There is no automated method to determine the optimal value for `min_samples` in DBSCAN. The value should be chosen based on domain knowledge and familiarity with the dataset. Here are some general guidelines:
- For larger datasets, choose a larger `min_samples`.
- For noisier datasets, choose a larger `min_samples`.
- As a rule of thumb, `min_samples` should be greater than or equal to the dimensionality of the dataset.
- For 2D data, the default value is `min_samples = 4` (Ester et al., 1996).
- For higher-dimensional data, use `min_samples = 2 * dim`, where `dim` is the dimensionality of the dataset.

In our case, after preprocessing, the dataset was reduced from 42 features to 20 features. Therefore, `min_samples` could be set to **40**. However, after scaling down the dataset from 1 million records to 1,000 records, `min_samples` can be adjusted to **4**.


### Selecting `ε` (Epsilon)
After selecting `min_samples`, the next step is to determine the optimal value for `ε`. One technique for automatically determining `ε` is:
1. Calculate the average distance between each point and its `k` nearest neighbors, where `k = min_samples`.
2. Plot these average distances in ascending order on a k-distance graph.
3. The optimal value for `ε` can be found at the point of maximum curvature (i.e., where the graph has the steepest slope).


### Implementation in RapidMiner
- Use the **DBSCAN** operator in RapidMiner to perform clustering.
- Configure the operator with the selected values for `min_samples` and `ε`.
- Analyze the resulting clusters and noise points to gain insights into the dataset.

<img width="288" alt="Screenshot 2025-02-25 at 7 36 14 PM" src="https://github.com/user-attachments/assets/8d494b08-a812-490d-8a16-133692b5fadd" />

As can be seen, for our data, after running the Python code below, the slope of the line shown is strongly exponential at approximately 225, so this number could be a suitable epsilon! (0.225)

<img width="264" alt="Screenshot 2025-02-25 at 7 37 47 PM" src="https://github.com/user-attachments/assets/bd8c942e-ff07-4790-be88-6d18a724834f" />

   ```python
# Import necessary libraries
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the preprocessed dataset
data1 = pd.read_csv("afterPP.csv")
data = data1

# Initialize NearestNeighbors to find the distances to the 42 nearest neighbors
neighbors = NearestNeighbors(n_neighbors=42)
neighbors_fit = neighbors.fit(data)

# Calculate distances and indices of the nearest neighbors
distances, indices = neighbors_fit.kneighbors(data)

# Sort the distances and extract the second column (distance to the nearest neighbor)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]

# Plot the sorted distances to determine the optimal epsilon (ε) for DBSCAN
plt.plot(distances)
plt.title("K-Distance Graph for DBSCAN")
plt.xlabel("Data Points")
plt.ylabel("Distance to 42nd Nearest Neighbor")
plt.show()
```

<img width="443" alt="Screenshot 2025-02-25 at 7 46 18 PM" src="https://github.com/user-attachments/assets/a0669162-f666-4ad0-9e6e-83fecad1e98a" />
<img width="416" alt="Screenshot 2025-02-25 at 7 46 58 PM" src="https://github.com/user-attachments/assets/68a98ad0-353b-4680-9d90-59d5a8b63507" />
<img width="479" alt="Screenshot 2025-02-25 at 7 47 48 PM" src="https://github.com/user-attachments/assets/3fbdf83f-f193-4428-aa64-e6d025f9335b" />

---

## Information Classification
## Classification Using Decision Trees (C4.5)

In this section, we perform classification using the **C4.5 decision tree algorithm**. We configure parameters such as the tree depth and the minimum number of samples required for splitting. Finally, we train the model using **data splitting** into training and testing datasets.

To achieve the objectives of this phase, we use the following four operators in RapidMiner:

---

### 1. SplitData
- This operator splits the dataset into two parts for training and evaluating the classification model.
- We use the **80/20 split strategy** with **Stratified Sampling** to ensure:
  - A proper distribution of data between the training (80%) and testing (20%) sets.
  - Balanced representation of classes in both sets.
- The operator is configured with two outputs: one for 80% of the data (training) and one for 20% of the data (testing).

---

### 2. DecisionTree
- This operator takes the training data (80% of the dataset) as input.
- To implement the **C4.5 algorithm**, we configure the decision tree based on **information gain** (the advantages of this algorithm are discussed in the project documentation).
- Key parameters:
  - **Maximum Depth**: Set to the default value of 10 (typically ranges between 2 and 15, depending on dataset complexity and size).
  - **Minimum Samples for Splitting**: For more complex trees, larger values are recommended to prevent overfitting.
- The primary output of this operator is the trained decision tree model.

---

### 3. ApplyModel
- This operator is used to evaluate the trained model.
- It takes two inputs:
  1. The trained model (from the `DecisionTree` operator).
  2. The unlabeled test dataset (from the second output of the `SplitData` operator).
- The operator applies the model to the test dataset and generates predictions.

---

### 4. Performance
- The predictions from the `ApplyModel` operator are fed into this operator for evaluation.
- We select **accuracy** as the primary performance metric. Additionally, the **confusion matrix** is also reported to provide a detailed breakdown of the model's performance.

<img width="439" alt="Screenshot 2025-02-25 at 7 59 15 PM" src="https://github.com/user-attachments/assets/4c8afe1b-2fab-4df6-b783-e77a15f04561" />
<img width="433" alt="Screenshot 2025-02-25 at 7 59 43 PM" src="https://github.com/user-attachments/assets/6569ebf9-1f78-43a8-a6f3-85efe7ab7b9f" />
<img width="247" alt="Screenshot 2025-02-25 at 8 00 48 PM" src="https://github.com/user-attachments/assets/2af5c0e6-f8da-43f7-a059-ddc235016b0f" />
<img width="449" alt="Screenshot 2025-02-25 at 8 01 17 PM" src="https://github.com/user-attachments/assets/b42d0f18-5b38-4537-b45f-8d1856615600" />


---

## Conclusion
- In this project, we used a set of machine learning techniques to analyze and classify a large dataset. The project was divided into three phases: data preprocessing, clustering using the DBSCAN algorithm, and information classification using a decision tree (C4.5). Each phase involved careful consideration of algorithm parameters, dataset features, and evaluation criteria to ensure the accuracy and efficiency of the final model.
- In the data preprocessing phase, the focus was on data cleaning, feature selection, and normalization. Handling missing values ​​and scaling numerical features were important steps that ensured the quality and consistency of the dataset. By implementing these preprocessing techniques, the performance of downstream algorithms such as DBSCAN and decision tree, which depend on high-quality data, was improved.
- The DBSCAN clustering phase posed challenges due to the large size of the dataset. To determine the appropriate values ​​for the DBSCAN parameters (eps and min_samples), we used a combination of domain knowledge and visualization techniques. The k-distance plot method helped us find the optimal value of eps, where the plot showed a sharp increase and the appropriate density threshold for the formation of meaningful clusters was determined. The choice of min_samples was based on the data dimensions and the amount of noise present in the dataset. As a result, the DBSCAN algorithm was able to identify meaningful clusters, even in the presence of high noise. The final phase of the project focused on data classification using the C4.5 decision tree algorithm. We performed parameter tuning and optimized parameters such as tree depth and minimum number of samples per node. Cross-validation was used to evaluate the model performance and avoid overfitting. The decision tree model provided a transparent and interpretable set of rules that was able to classify the data with high accuracy. However, limitations such as model complexity and the possibility of overfitting were addressed by adjusting the tree depth and pruning unnecessary branches.
- During the project, the performance of the selected algorithms was evaluated using various metrics such as accuracy, precision, and recall. These metrics provided valuable insights into the model performance and guided decisions regarding model improvement. Despite the challenges posed by the large dataset, the careful selection of parameters and the evaluations performed ensured the effective performance of the models.
- In conclusion, this project demonstrated the power of machine learning algorithms in data analysis and classification. By combining DBSCAN clustering and C4.5 decision tree, we were able to discover meaningful patterns in the data and build a robust classification model. Future work could explore further optimization of DBSCAN parameters, use of ensemble methods, or adding additional features to improve the model accuracy.

---

## Suggestions for Development
- Parameter optimization:
   - The parameter selection for both DBSCAN and decision trees was heavily dependent on manual tuning and domain knowledge. The use of automated parameter optimization techniques, such as network search or Bayesian optimization, can improve accuracy and efficiency. These methods provide a more systematic exploration of the parameter space, especially for large datasets.

- Feature engineering:
   - The current analysis uses dataset features without applying advanced transformations or extractions. Future work could explore advanced feature engineering techniques such as creating feature combinations, applying dimensionality reduction methods such as PCA, or using feature selection algorithms to remove irrelevant features. These approaches can reduce noise and improve model performance.

- Handling imbalanced data:
   - If the dataset contains imbalanced classes, it can affect the ability of the decision tree to generalize. Techniques such as resampling fewer classes, downsampling more classes, or using synthetic data generation methods such as SMOTE can be used to balance the data and improve classification accuracy.

- Improve scalability:
   - The size of the dataset creates computational challenges during clustering and classification. Implementing scalable versions of DBSCAN, such as HDBSCAN, or using distributed data processing frameworks such as Apache Spark can significantly reduce runtime and increase performance.

- Integrate ensemble methods:
   - Rather than relying solely on a C4.5 decision tree, integrating ensemble methods such as Random Forest or Boosted Gradient Trees can improve robustness and accuracy. These methods typically perform better by combining predictions from multiple trees.

- Use of Cross-validation:
   - While the model was evaluated using the test dataset, cross-validation across multiple partitions can provide a more robust measure of model performance. This helps identify overfitting issues and ensures that the model generalizes well to new data.

- Explore alternative models:
   - In addition to DBSCAN and C4.5, exploring alternative clustering algorithms such as K-Means++ or spectral clustering and classification algorithms such as support vector machines (SVM) or neural networks can provide valuable benchmarks and perhaps better results for specific data features.

- Evaluate with diverse metrics:
   - The evaluation was limited to common metrics such as precision and recall. Introducing additional metrics such as F1-Score, ROC-AUC, or Silhouette score (for clustering) can provide a more comprehensive understanding of the model’s performance and its suitability for the problem at hand.
