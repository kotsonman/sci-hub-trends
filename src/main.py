from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.utils.data import get_data_from_database
from src.utils.text import clean_text, tokenize_text

def vectorize_text(titles, max_features = 1000):
    """Векторизация текста с помощью TF-IDF."""
    vectorizer = TfidfVectorizer(max_features = max_features)
    vectorizer.fit(titles)
    vectors = vectorizer.transform(titles).toarray()
    return vectors, vectorizer


def cluster_titles(vectors, num_clusters = 5):
    """Кластеризация векторов с помощью KMeans."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(vectors)
    clusters = kmeans.labels_
    return clusters


def visualize_clusters(vectors, clusters, titles):
    """Визуализация кластеров (опционально)."""
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    df = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
    df['cluster'] = clusters
    df['title'] = titles
    plt.figure(figsize=(10, 8))
    for cluster in np.unique(clusters):
        subset = df[df['cluster'] == cluster]
        plt.scatter(subset['x'],subset['y'], label=f'Cluster {cluster}')
        for i, row in subset.iterrows():
            plt.annotate(row['title'], (row['x'], row['y']), fontsize = 8, alpha = 0.5)
    plt.title('Clusters of Articles Titles')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()

def get_cluster_keywords(vectorizer, vectors, clusters, titles, n_keywords = 5):
    """Определение ключевых слов для каждого кластера."""
    keywords = {}
    for cluster_num in np.unique(clusters):
        cluster_vectors = vectors[clusters == cluster_num]
        centroid = np.mean(cluster_vectors, axis=0)
        top_indices = np.argsort(centroid)[-n_keywords:]
        top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        cluster_titles = [title for title, cluster in zip(titles, clusters) if cluster == cluster_num]
        keywords[cluster_num] = {
            'keywords': top_keywords,
            'titles': cluster_titles
        }
    return keywords


def main():
    db_path = 'your_database.db'  # Замените на путь к вашей БД
    titles = get_data_from_database(db_path)

    # Предобработка текста
    cleaned_titles = [clean_text(title) for title in titles]
    tokenized_titles = [tokenize_text(title) for title in cleaned_titles]

    # Векторизация текста
    vectors, vectorizer = vectorize_text(tokenized_titles)

    # Кластеризация
    clusters = cluster_titles(vectors)

    # Визуализация (опционально)
    visualize_clusters(vectors, clusters, titles)

    # Ключевые слова для кластеров
    cluster_keywords = get_cluster_keywords(vectorizer, vectors, clusters, titles)
    for cluster_num, data in cluster_keywords.items():
        print(f"Cluster {cluster_num}:")
        print(f"  Keywords: {', '.join(data['keywords'])}")
        print(f"  Titles: {data['titles']}")
        print("-" * 20)

if __name__ == '__main__':
    main()
