# Deep Learning for Recipe Similarity: A Mathematical Analysis

## Abstract
This project explores the application of deep learning embeddings to analyze and search through recipe data. Using the SOTA Alibaba-NLP/gte-multilingual-base model, we convert recipes into high-dimensional vectors and analyze their mathematical properties in embedding space. Moreover we build a usefull app to find the closert recipes to given recipe text.

## 1. Introduction

### 1.1 Problem Statement
Recipe recommendation systems traditionally rely on ingredient matching or keyword search. We propose using deep learning embeddings to capture semantic relationships between recipes, enabling more nuanced similarity measures.

### 1.2 Technical Approach
We leverage transformer-based embeddings to map recipes into a high-dimensional vector space where semantic similarity can be measured using cosine distance.

## 2. Methodology

### 2.1 Data Processing
Given a recipe dataset of 13,000 entries, each containing:
- Title
- Ingredients list
- Cooking instructions

The data processing pipeline:


### 2.2 Mathematical Framework

#### Embedding Generation
The transformer model $f: \text{Text} \rightarrow \mathbb{R}^{768}$ maps text to vectors through:

1. Tokenization: $\text{text} \rightarrow \{t_1,...,t_n\}$
2. Contextual encoding: $\{t_1,...,t_n\} \rightarrow \{h_1,...,h_n\}$ 
3. [CLS] token pooling: $h_{[CLS]} \in \mathbb{R}^{768}$

#### Similarity Metric
For recipes $a,b$ with embeddings $x_a,x_b$:

$\text{similarity}(a,b) = \cos(x_a,x_b) = \frac{x_a \cdot x_b}{\|x_a\|\|x_b\|}$

## 3. Analysis
![kmeans_clusters.png](/images/kmeans_clusters.png)
![pairplot.png](/images/pairplot.png)
![pca_explained_variance.png](/images/pca_explained_variance.png)


[![Everything Is AWESOME](https://i.sstatic.net/q3ceS.png)](https://www.youtube.com/watch?v=ujC34WyIu_A "Everything Is AWESOME")


<div align="left">
      <a href="https://www.youtube.com/watch?v=5yLzZikS15k">
         <img src="https://img.youtube.com/vi/5yLzZikS15k/0.jpg" style="width:100%;">
      </a>
</div>

### 3.1 Dimensionality Analysis
Principal Component Analysis reveals:

```python
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
```

- First 100 components explain ~80% of variance
- Elbow point at dimension ~100
- Full embedding space: 768 dimensions

### 3.2 Clustering Analysis
K-means clustering with $k \in \{2,3,5,10\}$ shows:

$J(C) = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$

Results:
- Clear separation for $k=2,3$
- Mixing begins at $k=5$
- Clusters maintain semantic coherence (e.g., desserts vs. main dishes)

### 3.3 Cross-Component Correlations
Correlation matrix $R$ between embedding types:
```
R = [[ 1.   0.3  0.4  0.6],
     [0.3   1.  0.35 0.7],
     [0.4  0.35  1.  0.9],
     [0.6  0.7   0.9  1.]]
```
- Strong correlation (0.9) between full recipe and instructions
- Weak correlation (0.3) between titles and ingredients

## 4. Implementation

### 4.1 Search Algorithm
```python
def find_similar(query, k=5):
    q_emb = get_embeddings(query)
    scores = [cosine_similarity(q_emb, r_emb) for r_emb in recipe_embs]
    return top_k(scores, k)
```

### 4.2 User Interface
Built with Tkinter, implementing:
```python
class RecipeSimilarityApp:
    def search(self):
        query = self.search_entry.get()
        similar = find_similar(query)
        self.display_results(similar)
```

## 5. Results

### 5.1 Quantitative Metrics
- Average query time: 0.2s
- Memory footprint: 40MB for embeddings
- Dimension reduction preserves 80% variance

### 5.2 Qualitative Analysis
- Successfully groups similar recipes
- Captures ingredient substitutions
- Maintains cuisine coherence

## 6. Conclusion & Future Work

### 6.1 Key Findings
1. Recipe semantics well-captured in ~100 dimensions
2. Instructions dominate semantic meaning
3. Natural clustering emerges in embedding space

### 6.2 Future Directions
1. Cross-lingual recipe matching
2. Ingredient-aware embeddings
3. Hierarchical clustering

## References
1. Alibaba-NLP/gte-multilingual-base documentation
2. Transformer architectures for text embeddings
3. Recipe embedding papers

[View full implementation on GitHub](https://github.com/username/recipe-embeddings)