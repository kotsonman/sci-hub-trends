import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer

# Импорт изменен, чтобы исключить sys.path.append
from utils.data import get_data_from_database
from utils.text import clean_text, tokenize_text

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def vectorize_text(titles, max_features):
    """Векторизация текста с помощью TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(titles)
    vectors = vectorizer.transform(titles).toarray()
    return vectors, vectorizer


def cluster_titles(vectors, num_clusters, random_state=42, n_init=10):
    """Кластеризация векторов с помощью KMeans."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=n_init)
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
        plt.scatter(subset['x'], subset['y'], label=f'Cluster {cluster}')
    plt.title('Clusters of Articles Titles')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()


def get_cluster_keywords(vectorizer, vectors, clusters, titles, n_keywords):
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


def translate_text(texts, translator_model, translator_tokenizer, device, batch_size=32):
    """Перевод текста с использованием MarianMT."""
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            with torch.inference_mode():
                inputs = translator_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}  # Перемещаем входные данные на GPU
                outputs = translator_model.generate(**inputs, max_new_tokens=100)
                translated_batch = [translator_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
                translated_texts.extend(translated_batch)
        except Exception as e:
            logging.error(f"Error translating batch: {batch}. Error: {e}")
            translated_texts.extend(batch)  # Сохраняем исходный текст при ошибке
    return translated_texts


def get_cluster_name(cluster_titles, cluster_keywords, tokenizer, model, device, num_titles=10):
    """Определение имени кластера с помощью Flan-T5."""
    try:
        # Случайный выбор заголовков
        sample_titles = np.random.choice(cluster_titles, size=min(num_titles, len(cluster_titles)), replace=False)

        # Добавляем ключевые слова в промпт
        keywords_str = ", ".join(cluster_keywords)
        prompt = f"Define the topic of a cluster of scientific articles based on the following titles and keywords: {', '.join(sample_titles)}. Keywords: {keywords_str}"

        with torch.inference_mode():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)  # Перемещаем входные данные на GPU
            outputs = model.generate(input_ids, max_new_tokens=70)
            cluster_name = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Пост-обработка для удаления дубликатов
        cluster_name = " ".join(list(dict.fromkeys(cluster_name.split())))
        return cluster_name
    except Exception as e:
        logging.error(f"Error in get_cluster_name: {e}")
        return "Error generating name"

def find_optimal_batch_size(titles, translator_model, translator_tokenizer, device):
    """Подбор оптимального размера пакета."""
    best_batch_size = 128  # Начальное значение
    return best_batch_size

def main(db_path, output_file, num_clusters, max_features, n_keywords, visualize, num_records=14000):
    """Основная функция для кластеризации и анализа статей."""

    # Определяем устройство (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Проверяем, есть ли уже файл с результатами
    if os.path.exists(output_file):
        logging.info(f"Loading cluster results from {output_file}...")
        with open(output_file, "r") as f:
            results = json.load(f)
            for cluster_name, cluster_data in results.items():
                print(f"Cluster {cluster_name}:")
                print(f"  Keywords: {', '.join(cluster_data['keywords'])}")
                print("-" * 20)
        return

    titles = get_data_from_database(db_path)
    if not titles:
        logging.error("No titles found in the database.")
        return

    # Ограничиваем количество записей
    titles = titles[:num_records]
    logging.info(f"Processing {len(titles)} titles.")

    # Загрузка модели и токенизатора для перевода
    translator_model_name = "Helsinki-NLP/opus-mt-ru-en"
    translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
    translator_model = MarianMTModel.from_pretrained(translator_model_name).to(device)  # Перемещаем модель на GPU

    # Подбор оптимального размера пакета
    optimal_batch_size = find_optimal_batch_size(titles, translator_model, translator_tokenizer, device)

    # Перевод заголовков на английский
    logging.info("Translating titles to English...")
    translated_titles = translate_text(titles, translator_model, translator_tokenizer, device, optimal_batch_size)

    # Предобработка текста
    cleaned_titles = [clean_text(title) for title in translated_titles]
    tokenized_titles = [tokenize_text(title) for title in cleaned_titles]

    # Векторизация текста
    vectors, vectorizer = vectorize_text(tokenized_titles, max_features)

    # Кластеризация
    clusters = cluster_titles(vectors, num_clusters=num_clusters)

    # Визуализация (опционально)
    if visualize:
        visualize_clusters(vectors, clusters, translated_titles)

    # Ключевые слова для кластеров
    cluster_keywords = get_cluster_keywords(vectorizer, vectors, clusters, translated_titles,
                                            n_keywords)  # Передаем translated_titles

    # Загрузка модели и токенизатора для генерации имен кластеров
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)  # Перемещаем модель на GPU

    results = {}
    for cluster_num, data in cluster_keywords.items():
        cluster_name = get_cluster_name(data['titles'], data['keywords'], tokenizer, model, device)
        results[cluster_name] = {
            "keywords": data['keywords'],
            "titles": data['titles']
        }
        print(f"Cluster {cluster_name}:")
        print(f"  Keywords: {', '.join(data['keywords'])}")
        print("-" * 20)

    # Сохраняем результаты в файл
    print(f"Saving cluster results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster articles titles.")
    parser.add_argument("--db_path", type=str, default="../data/my_database.db", help="Path to the database file.")
    parser.add_argument("--output_file", type=str, default="cluster_results.json", help="Path to the output JSON file.")
    parser.add_argument("--num_clusters", type=int, default=7, help="Number of clusters.")
    parser.add_argument("--max_features", type=int, default=1000, help="Max features for TF-IDF.")
    parser.add_argument("--n_keywords", type=int, default=5, help="Number of keywords per cluster.")
    parser.add_argument("--visualize", action="store_true", help="Visualize clusters.")

    args = parser.parse_args()

    main(args.db_path, args.output_file, args.num_clusters, args.max_features, args.n_keywords, args.visualize)