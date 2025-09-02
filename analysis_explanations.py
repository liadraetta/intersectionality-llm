import argparse
import numpy as np
import pandas as pd
from utils.embeddings_clustering import *
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from sklearn import metrics
from glob import glob
import os
import re
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# ---------------------------
# ARGPARSE
# ---------------------------

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=["positive", "negative", "overall"], default="overall")
    parser.add_argument('--len_token', action='store_true', help='compute token length')
    return parser.parse_args()

args = parse_command_line_args()
filter_type = args.filter
len_token = args.len_token

# ---------------------------
# LOAD DATA
# ---------------------------

dataset = pd.read_csv("dataset/AnnAttDataset.csv")
baseline = pd.read_csv("results/predictions/cleaned/cleaned_predictions_Llama_noCoT_baseline.csv")

baseline["reasoning"] = baseline["parsed_output"].str.extractall(r'\[([^]]+)\]').groupby(level=0)[0].last()
baseline = baseline.drop_duplicates(subset=["postId", "reasoning"])
baseline["source"] = "baseline"

baseline = baseline[["postId", "annId", "offensiveYN", "reasoning", "source", "prediction"]]


# Apply offensiveYN filter to baseline
if filter_type == "positive":
    baseline = baseline[baseline["prediction"] == 1]
elif filter_type == "negative":
    baseline = baseline[baseline["prediction"] == 0]

print("baseline shape: ", baseline.shape, "with filter_type ", filter_type)


# ---------------------------
# PROCESS DEMOGRAPHIC FILES
# ---------------------------
dir = "results/predictions_dem/Llama/cleaned"
file_pattern = "cleaned_predictions_Llama_noCoT_*.csv"
    
for file in glob(os.path.join(dir, file_pattern)):
    filename = file.split("/")[-1]
    input_name = re.search(r'(?<=noCoT_)[^.]*', filename).group()
        
    print(f"\nProcessing: {input_name}\n")
        
    demographics = pd.read_csv(file)
    demographics["reasoning"] = demographics["parsed_output"].str.extractall(r'\[([^]]+)\]').groupby(level=0)[0].last()
    demographics = demographics.drop_duplicates(subset=["postId", "reasoning"])
    demographics["source"] = input_name

    demographics = demographics[["postId", "annId", "offensiveYN", "reasoning", "source", "prediction"]]

    # Apply offensiveYN filter to demographics
    if filter_type == "positive":
        demographics = demographics[demographics["prediction"] == 1]
        path = "results/output_expl_analysis/positive"
    elif filter_type == "negative":
        demographics = demographics[demographics["prediction"] == 0]
        path = "results/output_expl_analysis/negative"
    elif filter_type == "overall":
        path = "results/output_expl_analysis/overall"
    else: 
        raise ValueError("choose a filter_type among the following ['positive', 'negative', 'overall']")


    print(f"{input_name} shape: {demographics.shape} with filter_type {filter_type}")


    # merge with baseline and dataset
    df = pd.concat([baseline, demographics], axis=0)
    df = df.drop_duplicates(subset=["postId", "reasoning", "source"])
    print(f"df shape: {df.shape} with filter_type {filter_type}")

    df = df.merge(dataset, on=["postId", "annId", "offensiveYN"])
    print(f"df shape after merging with full dataset: {df.shape} with filter_type {filter_type}\n")


    # Load or compute embeddings
    try:
        final_embeddings = np.load(f"{path}/embeddings/embedding_{input_name}.npy")
        print(f"Loaded existing {input_name} embeddings")
    except FileNotFoundError:
        print(f"Computing {input_name} embeddings...")
        final_embeddings = BERT_embeddings(df, "reasoning")
        np.save(f"{path}/embeddings/embedding_{input_name}.npy", final_embeddings)
        
    # ---------------------------
    # PCA
    # ---------------------------    
    # pca = PCA()
    # pca.fit(final_embeddings)
    # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # n_components = np.argmax(cumulative_variance >= 0.90) + 1
    # pca_final = PCA(n_components=n_components)
    # reduced_embeddings = pca_final.fit_transform(final_embeddings)
    # print(f"Number of components chosen: {n_components}")
 



    # ---------------------------
    # CLUSTERING
    # ---------------------------   
    model_kmeans, k_best = obtain_kmeans_model(final_embeddings)


    labels = model_kmeans.labels_
    df["cluster"] = labels 

    
        
    print(f"=== {input_name} Results ===")
    print(f"Silhouette Coefficient: {metrics.silhouette_score(final_embeddings, model_kmeans.labels_):.3f}")
    print(f"Calinski-Harabasz Index: {metrics.calinski_harabasz_score(final_embeddings, model_kmeans.labels_):.3f}")
    print(f"Davies-Bouldin Index: {metrics.davies_bouldin_score(final_embeddings, model_kmeans.labels_):.3f}")





    # ---------------------------
    # TSNE
    # --------------------------- 
    if filter_type == "positive":
        tsne_output(final_embeddings, model_kmeans, input_model=input_name, dir=path, color="Dark2")
    elif filter_type == "negative":
        tsne_output(final_embeddings, model_kmeans, input_model=input_name, dir=path, color="RdBu")
    elif filter_type == "overall":
        tsne_output(final_embeddings, model_kmeans, input_model=input_name, dir=path, color="RdYlBu")


    df.to_csv(f"{path}/dataframes/clusters_{input_name}.csv", index=False)





    results_path = f"{path}/results/results_{input_name}.txt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:  # use "w" to overwrite or create a new file each time
        f.write(f"=== Results for {input_name} ===\n")
        f.write(f"Silhouette Coefficient: {metrics.silhouette_score(final_embeddings, model_kmeans.labels_):.3f}\n")
        f.write(f"Calinski-Harabasz Index: {metrics.calinski_harabasz_score(final_embeddings, model_kmeans.labels_):.3f}\n")
        f.write(f"Davies-Bouldin Index: {metrics.davies_bouldin_score(final_embeddings, model_kmeans.labels_):.3f}\n")
        f.write("\nCluster vs Source:\n")
        f.write(pd.crosstab(df["cluster"], df["source"]).to_string())
        f.write("\n" + "-"*50 + "\n")
        f.write("Cluster vs OffensiveYN:\n")
        f.write(pd.crosstab(df["cluster"], df["offensiveYN"]).to_string())
        f.write("\n" + "-"*50 + "\n")
        f.write("Cluster vs prediction:\n")
        f.write(pd.crosstab(df["cluster"], df["prediction"]).to_string())
        
        if filter_type == "positive" or filter_type == "negative":
            f.write("\n" + "-"*50 + "\n")
            f.write(pd.crosstab(df["cluster"], df["isAAE"]).to_string())
            f.write("\n" + "-"*50 + "\n")
            f.write(pd.crosstab(df["cluster"], df["vulgar"]).to_string())
            f.write("\n" + "-"*50 + "\n")
            f.write(pd.crosstab(df["cluster"], df["targetsBlackPeople"]).to_string())
        
        f.write("\n" + "="*100 + "\n")



one_plot(filter_type=filter_type)

if len_token:
    kde_plot(filter_type=filter_type)
