# Description
Understanding human personality plays a crucial role across numerous fields such as psychology, behavioral science, sociology, human behavior analysis, human-computer interaction, automated personality detection, leadership studies, and many others.
In order to understand personality, traditionally it is broken down into a handful of dimensions (or bases) like the Big-5 OCEAN traits, or the NEO traits, where an individual is assessed on all the bases. The way these bases have been agreed upon has been rigorous but has not been revisited for decades.
In this work, we use newer AI methodologies like Large Language Models for word-base representations and K-Means and Neural Networks for understanding the underlying bases, in order to assess whether newer techniques agree with the widely accepted personality bases. We bring new insights by incorporating the older studies with the newer methodologies. 
Our analysis reveal that even though there is some agreement between the OCEAN traits and the bases we discover. The number of bases optimal for dividing people into groups is 2, however 5 is the next best number. The actual bases we discover agree somewhat with the Big-5 OCEAN traits, with some proposed modifications. 

# Code details 
1. analyse_kmeans.py
   - analyses the kmeans models and analyses cluster representative for the clusters. 
2. cluster.py
   - this file read words from the file selected (e.g. norman_75.txt). It then extracts embeddings from a model (either bert, llama, t5, opt, flant5).
   - Runs Kmeans and saves the extracted embeddings
3. mlp.py
   - reads the saved embeddings and trains a simple MLP model based on the defined loss
4. run_umap.py
   - loads embeddings and visualizes clusters

# Findings 
## Cluster Entropy for Models (KMeans vs MLP)
![image](https://github.com/user-attachments/assets/d2ded505-f08c-48bc-8481-28ab5c8f1ae9)

1. MLP model are able to perform better clustering than KMeans is able to, by observing that the cluster purity is lower for mlp models compared to the kmeans model. 
2, We can also observe that the trend of the cluster entropy is similar across the mlp models, where the lowest cluster entropy is when the number of clusters are 2, 5, 6 and 8.

## Cluster Entropy vs Cluster size:
![cluster_purity_across_configs](https://github.com/user-attachments/assets/d3e9fc1c-2d59-4e53-8b5e-c25192619a64)

## Representative Work for cluster size=2 

![image](https://github.com/user-attachments/assets/f4590960-5a30-431a-934e-744c20998d22)

# Find our paper here: 
tbd

# Cite our work 
tbd
