import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.cluster import KMeans

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

dataset = pd.read_csv('data/13k-recipes.csv', index_col=0)
dataset['Ingredients'] = dataset['Ingredients'].apply(lambda x: " ".join(eval(x)))
dataset.fillna('', inplace=True)


def reduce_dimensions(embeddings_list, method='pca', n_components=2):
    embeddings_list_length = [len(embeddings) for embeddings in embeddings_list]
    embeddings = np.concatenate(embeddings_list, axis=0)
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError('Invalid method')
    embeddings_red = reducer.fit_transform(embeddings)
    embeddings_red_list = []
    i = 0
    for length in embeddings_list_length:
        embeddings_red_list.append(embeddings_red[i:i + length])
        i += length
    return embeddings_red_list


title_embeddings = np.load('data/title_embeddings.npy')
ingredients_embeddings = np.load('data/ingredients_embeddings.npy')
instructions_embeddings = np.load('data/instructions_embeddings.npy')
all_embeddings = np.load('data/all_embeddings.npy')

title_1d, ingredients_1d, instructions_1d, all_1d = reduce_dimensions(
    [title_embeddings, ingredients_embeddings, instructions_embeddings, all_embeddings],
    method='pca',
    n_components=1
)

df = pd.DataFrame({
    'Title': title_1d.flatten(),
    'Ingredients': ingredients_1d.flatten(),
    'Instructions': instructions_1d.flatten(),
    'All': all_1d.flatten()
})

# Create pairplot
sns.pairplot(df)
plt.savefig('pairplot.png', dpi=300)
plt.show()

# tsne_2d = TSNE(n_components=2).fit_transform(all_embeddings)
#
# plt.figure(figsize=(8, 6))
# plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], alpha=0.5)
# plt.title('TSNE 2D visualization of All Embeddings')
# plt.show()

# 2. K-means clustering and PCA visualization
from sklearn.cluster import KMeans

# K-means clustering on title embeddings
k_values = [2, 3, 5, 10]
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

for idx, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(title_embeddings)

    # Plot in 2D PCA space
    pca_2d = PCA(n_components=2).fit_transform(title_embeddings)
    row, col = idx // 2, idx % 2
    axs[row, col].scatter(pca_2d[:, 0], pca_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    axs[row, col].set_title(f'K-means (k={k})')

    # Print sample titles from each cluster
    print(f"\nK={k} clusters:")
    for cluster in range(k):
        cluster_indices = np.where(clusters == cluster)[0]
        sample_indices = np.random.choice(cluster_indices, min(10, len(cluster_indices)), replace=False)
        print(f"\nCluster {cluster}:")
        for idx in sample_indices:
            print(f"- {dataset['Title'].iloc[idx]}")

plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=300)
plt.show()

# Show explained variance ratio in PCA
pca = PCA().fit(all_embeddings)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.savefig('pca_explained_variance.png', dpi=300)
plt.show()
