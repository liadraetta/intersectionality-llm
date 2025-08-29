from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import nltk
nltk.download("punkt_tab")


def BERT_embeddings(df, col_text):
    print("Generating BERT embeddings...")
    from transformers import BertTokenizer, BertModel
    import torch
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Move model to GPU
    model.to(device)
    
    texts = df[col_text].tolist()
    embeddings = []
    
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
                            
            # Get CLS token embedding and move to CPU for numpy conversion
            embedding = outputs['last_hidden_state'][:, 0, :].cpu().numpy()
            embeddings.append(embedding)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(texts)} texts")
    
    print("BERT embeddings created.")
    emb_npy = np.array(embeddings)
    final_embeddings = np.reshape(emb_npy, (emb_npy.shape[0], emb_npy.shape[2]))
    return final_embeddings



def intrinsic_evaluation (model, final_embeddings, labels):
    print("Dendrogram sklearn estimated number of clusters: ", model.n_clusters_)
    print(f"Silhouette Coefficient: {metrics.silhouette_score(final_embeddings, labels):.3f}")
    print(f"Calinski-Harabasz Index: {metrics.calinski_harabasz_score(final_embeddings, labels):.3f}")
    print(f"Davies-Bouldin Index: {metrics.davies_bouldin_score(final_embeddings, labels):.3f}")



def obtain_kmeans_model (final_embeddings): 
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from warnings import simplefilter
    from tqdm import tqdm

    simplefilter("ignore", ConvergenceWarning)

    k_range = range(2, 10)
    best_k = {}
    for k in tqdm(k_range):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        model.fit(final_embeddings)
        labels = model.labels_
        try:
            score = metrics.silhouette_score(final_embeddings, labels)
        except ValueError:
            score = -1
        best_k[k] = score

    # Select best number of clusters
    k_best = max(best_k, key=best_k.get)
    print(f"Best KMeans clusters: {k_best}")

    # Final model
    model_kmeans = KMeans(n_clusters=k_best, random_state=42, n_init='auto')
    model_kmeans.fit(final_embeddings)

    return model_kmeans, k_best








def tsne_output(final_embeddings, model_kmeans, input_model, dir, color):
    # Step 1: Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='euclidean')
    X_2d = tsne.fit_transform(final_embeddings)

    # Step 2: Get cluster labels
    labels = model_kmeans.labels_
    n_clusters = model_kmeans.n_clusters

    # Step 3: Create color map
    cmap = plt.cm.get_cmap(color, n_clusters)  # color should be a string like 'spring', 'cool', 'winter'
    colors = [cmap(i) for i in range(n_clusters)]

    # Step 4: Plot
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(
            X_2d[labels == i, 0],
            X_2d[labels == i, 1],
            label=f'Cluster {i}',
            alpha=0.7,
            color=colors[i]
        )

    plt.title(f'{input_model}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dir}/tsne_plot/cluster_{input_model}.png")
    plt.close()






def one_plot(filter_type):
    import matplotlib.image as mpimg
    import os
    folder = f"output_expl_analysis/{filter_type}"
    tsne_dir = f"{folder}/tsne_plot"

    files = [
        "cluster_gender.png",
        "cluster_race.png",
        "cluster_political.png",
        "cluster_gender_race.png",
        "cluster_gender_political.png",
        "cluster_race_political.png",
        "cluster_gender_race_political.png"
    ]

    # Create 2 rows x 4 columns grid (1 empty slot)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flat):
        if i < len(files):
            img_path = os.path.join(tsne_dir, files[i])
            img = mpimg.imread(img_path)
            ax.imshow(img)
        ax.axis("off")  # hide ticks

    # Optionally, hide the last empty subplot
    if len(files) < axes.size:
        axes.flat[-1].axis("off")

    plt.tight_layout()
    plt.savefig(f"output_expl_analysis/plot_{filter_type}.png")
    plt.show()




def kde_plot(filter_type):
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    from glob import glob
    import os
    import matplotlib.pyplot as plt
    import pandas as pd 


    folder = f"output_expl_analysis/{filter_type}"
    dataframes_dir = f"{folder}/dataframes"
    pattern = "clusters_*.csv"
    for file in glob(os.path.join(dataframes_dir, pattern)):
        len_sent = []
        len_word = []
        df = pd.read_csv(file)
        for i,row in df.iterrows(): 
            text = row["reasoning"]
            sent = sent_tokenize(text)
            words = word_tokenize(text)

            len_sent.append(len(sent))
            len_word.append(len(words))

        df["len_sent"] = len_sent
        df["len_word"] = len_word

        filename = file.split("/")[-1].replace(".csv", "")

        with open(f"{folder}/kde_plot/len_word_{filename}.txt", "w") as f:  # use "w" to overwrite or create a new file each time
            f.write(f"=== Results for {filename} ===\n\n")
            
            f.write("MEAN\n")
            f.write(df.groupby("cluster")["len_word"].mean().to_string())
            f.write("\n\nSTD\n")
            f.write(df.groupby("cluster")["len_word"].std().to_string())
            f.write("\n\nMEDIAN\n")
            f.write(df.groupby("cluster")["len_word"].median().to_string())
            f.write("\n")
            f.write("-"*100)

        
        for cluster in df['cluster'].unique():
            subset = df[df['cluster'] == cluster]['len_word']
            subset.plot.kde(label=f'Cluster {cluster}')  # KDE gives smooth density curve

        plt.xlabel("Word Length")
        plt.ylabel("Density")
        plt.title(f"{filename}")
        plt.legend()
        plt.savefig(f"{folder}/kde_plot/{filename}.png")
        plt.close()