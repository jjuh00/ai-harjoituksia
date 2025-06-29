"""
Ohjelma, joka analysoi hotelliarvosteluja TripAdvisorista käyttäen luonnollisen kielen käsittelyä (NLP) ja koneoppimista.

Ohjelma esikäsittelee tekstit, suorittaa tunneanalyysin teksteistä, erittelee avainsanat, 
klusteroi arvostelut ja luo visualisoinnit analyysin tuloksista.
"""

import os
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATASET_FOLDER = "aineisto"
DATASET_FILE = "tripadvisor_hotel_reviews.csv"

def load_data(dataset_folder, dataset_file):
    """
    Lataa hotelliarvostelut CSV-tiedostosta.

    Parametrit:
        dataset_folder (str): Kansion polku, jossa CSV-tiedosto sijaitsee
        dataset_file (str): CSV-tiedoston nimi

    Palauttaa:
        DataFrame: Ladattu aineisto, joka sisältää hotelliarvosteluiden sarakkeet
    """
    try:
        df = pd.read_csv(os.path.join(dataset_folder, dataset_file))
        df = df.dropna()
        print(f"Aineisto ladattu onnistuneesti. Aineisto sisältää {len(df)} arvostelua")
        print(f"Sarakkeet: {list(df.columns)}")
        return df
    except FileNotFoundError:
        return None
    
def preprocess_text(text):
    """
    Esikäsittelee tekstin siistimällä ja normalisoimalla.
    
    Parametrit:
        text (str): Raaka teksti, joka halutaan esikäsitellä
        
    Palauttaa:
        str: Siistitty ja normalisoitu teksti
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()

    # Poistetetaan erikoismerkit ja numerot
    text = re.sub(r"[^a-zA-Z\s]", '', text)

    # Poistetaan yksittäiset kirjaimet ja lyhyet sanat (alle 3 kirjainta)
    text = re.sub(r"\b\w{1,2}\b", '', text)

    # Poistetaan ylimääräiset välilyönnit
    text = ' '.join(text.split())

    return text

def extract_keywords(texts):
    """
    Avainsanojen erittely NLTK:n luonnollisen kielen käsittelyn avulla.

    Parametrit:
        texts (list): Lista teksteistä, joista avainsanat halutaan eritellä

    Palauttaa:
        list: Lista tupleista (avainsana, esiintymistiheys)
    """
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("stopwords", quiet=True)
    except:
        print("NLTK-resurssien lataaminen epäonnistui. Varmista, että NLTK on asennettu")
        return []
    
    stop_words = set(stopwords.words("english"))
    all_keywords = []

    for text in texts:
        if not text:
            continue

        # Tokenisoidaan teksti
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        # Valitaan vain substantiivit ja adjektiivit
        keywords = [word.lower() for word, pos in tagged if pos.startswith(("NN", "JJ")) and 
                    word.lower() not in stop_words and len(word) > 2]
        all_keywords.extend(keywords)

    # Lasketaan ja palautetaan avainsanat
    keywords_count = Counter(all_keywords)
    return keywords_count.most_common(50)

def analyze_sentiment(text):
    """
    Analysoi tekstien tunteet TextBlobin avulla.

    Parametrit:
        text (str): Analysoitava teksti

    Palauttaa:
        str: Tunnekategoria ("positiivinen", "neutraali" tai "negatiivinen")
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "positiivinen"
    elif polarity < -0.1:
        return "negatiivinen"
    else:
        return "neutraali"
    
def vectorize_texts(texts):
    """
    Muuntaa tekstit TF-IDF-matriisiksi.
    
    Parametrit:
        texts (list): Lista käsitellyistä teksteistä
        
    Palauttaa:
        tuple: (TF-IDF-matriisi, TfidfVectorizer-olio)
    """
    vectorizer = TfidfVectorizer(
        max_features=400,
        stop_words="english",
        ngram_range=(1, 2),  # Säilytetään uni- ja digrammit
        min_df=0.25, # Poistetaan harvinaiset sanat
        max_df=0.75 # Poistetaan erittäin yleiset sanat
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer

def find_optimal_clusters(tfidf_matrix):
    """
    Etsii optimaalisen klusterien määrän silhouette-analyysin avulla.

    Parametrit:
        tfidf_matrix (spmatrix): Harva TF-IDF-matriisi

    Palauttaa:
        int: Optimaalinen klusterien määrä
    """
    silhouette_scores = []
    k_range = range(2, min(16, tfidf_matrix.shape[0]))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, cluster_labels)
        silhouette_scores.append(score)

    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimaalinen klusterien määrä: {optimal_k} (silhouette-pisteet: {max(silhouette_scores):.4f})")
        return optimal_k
    else:
        return 2 # Oletusarvo, jos ei löydy sopivaa klusterimäärää

def cluster_reviews(tfidf_matrix, n_clusters=None):
    """
    Klusteroi hotelliarvostelut KMeans-algoritmin avulla.

    Parametrit:
        tfidf_matrix (spmatrix): Harva TF-IDF-matriisi
        n_clusters (int): Klusterien määrä. Jos None, käytetään optimaalista klusterien määrää

    Palauttaa:
        tuple: (klusteritunnisteet, sovitettu KMeans-malli)
    """
    if n_clusters is None:
        n_clusters = find_optimal_clusters(tfidf_matrix)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    return cluster_labels, kmeans
    
def get_cluster_keywords(vectorizer, kmeans_model):
    """
    Hakee klustereiden avainsanat KMeans-mallin avulla.

    Parametrit:
        vectorizer (TfidfVectorizer): TfidfVectorizer-olio
        kmeans_model (KMeans): Sovitettu KMeans-malli

    Palauttaa:
        dict: Sanakirja klusterin avainsanoista
    """
    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for i, centroid in enumerate(kmeans_model.cluster_centers_):
        # Haetaan tämän klusterin top ominaisuuksien indeksit
        top_indices = centroid.argsort()[-5:][::-1]
        top_keywords = [feature_names[idx] for idx in top_indices]
        cluster_keywords[i] = top_keywords

    return cluster_keywords

def create_visualizations(df, cluster_labels, keywords):
    """
    Luo kattavan visualisoinnin arvosteluanalyysin tuloksista.

    Parametrit:
        df (DataFrame): Alkuperäinen DataFrame hotelliarvosteluista
        cluster_labels (ndarray): Klusteritunnisteet
        keywords (list): Suosituimmat avainsanat, jotka on eritelty teksteistä
    """
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(20, 16))

    # 1. Ympyrädiagrammi tunteista
    plt.subplot(2, 3, 1)
    sentiments = df["sentiment"].value_counts(normalize=True) * 100
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1"]
    plt.pie(sentiments.values, labels=sentiments.index, autopct=lambda p: '{:.1f}%'.format(p) if p > 1 else '', colors=colors, startangle=90)
    plt.title("Tunteiden jakauma", fontsize=16, fontweight="bold")

    # 2. Arvostelujen (arvosanojen) jakauma (histogrammi)
    plt.subplot(2, 3, 2)
    ratings = df["Rating"].value_counts().sort_index()
    bars = plt.bar(ratings.index, ratings.values, color="#96ceb4", alpha=0.8)
    plt.xlabel("Arvosana (1-5)")
    plt.ylabel("Arvostelujen määrä")
    plt.title("Arvostelujen jakauma", fontsize=16, fontweight="bold")
    plt.xticks(range(1, 6))

    # Lisätään arvot pylväisiin
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f"{int(height)}", ha="center", va="bottom")

    # 3. Top 5 avainsanaa (palkkikaavio)
    plt.subplot(2, 3, 3)
    top_5_keywords = keywords[:5]
    keyword_names = [keyword[0] for keyword in top_5_keywords]
    keyword_counts = [keyword[1] for keyword in top_5_keywords]

    bars = plt.barh(keyword_names, keyword_counts, color="#ffa07a", alpha=0.8)
    plt.xlabel("Esiintymistiheys")
    plt.title("Top 5 avainsanaa", fontsize=16, fontweight="bold")

    # Lisätään arvot palkkeihin
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2.,
                 f"{int(width)}", ha="left", va="center")

    # 4. Klustereiden jakauma (histogrammi) - yksinkertaistettu
    plt.subplot(2, 3, 4)
    cluster_counts = Counter(cluster_labels)
    plt.bar(cluster_counts.keys(), cluster_counts.values(), color="#dda0dd", alpha=0.8)
    plt.xlabel("Klusterinumero")
    plt.ylabel("Arvostelujen määrä")
    plt.title("Klustereiden jako", fontsize=16, fontweight="bold")
    plt.xticks(list(cluster_counts.keys()))

    # 5. Tunne vs arvosana lämpökartta - yksinkertaistettu
    plt.subplot(2, 3, 5)
    sentiment_rating = pd.crosstab(df["sentiment"], df["Rating"])
    sns.heatmap(sentiment_rating, annot=True, fmt='d', cmap="YlOrRd", cbar_kws={'label': 'Arvostelujen määrä'})
    plt.title("Tunteiden ja arvosanojen suhde", fontsize=16, fontweight="bold")
    plt.xlabel("Arvosana (1-5)")
    plt.ylabel("Tunne")

    plt.tight_layout(pad=3.0)
    plt.show()

def print_summary(df, cluster_labels, keywords, cluster_keywords):
    """
    Tulostaa yhteenvedon analyysin tuloksista.

    Parametrit:
        df (DataFrame): Alkuperäinen DataFrame hotelliarvosteluista
        cluster_labels (ndarray): Klusteritunnisteet
        keywords (list): Suosituimmat avainsanat
        cluster_keywords (dict): Klustereiden avainsanat
    """
    print("\nYhteenveto analyysistä:")
    print(f"Arvostelujen määrä: {len(df)}")
    print(f"Arvostelujen keskiarvosana: {df["Rating"].mean():.2f}")
    print(f"Arvosanojen jakauma: {', '.join(f"{k}: {v}" for k, v in df["Rating"].value_counts().sort_index().items())}")

    print(f"Tunneanalyysin tulokset:")
    sentiment_counts = df["sentiment"].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\nAvainsanat:")
    for i, (keyword, freq) in enumerate(keywords[:10], 1):
        print(f"{i:2d}. {keyword}: {freq}")

    print(f"\nKlusterointitulokset:")
    cluster_counts = Counter(cluster_labels)
    print(f"Klustereiden määrä: {len(cluster_counts)}")
    for cluster, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        print(f"Klusteri {cluster}: {count} arvostelua ({percentage:.1f}%)")

    print(f"\nKlustereiden suosituimmat avainsanat:")
    for cluster, keywords_list in cluster_keywords.items():
        print(f"Klusteri {cluster}: {', '.join(keywords_list)}")

def main():
    # Ladataan aineisto
    df = load_data(DATASET_FOLDER, DATASET_FILE)
    if df is None:
        return
    
    # Aineiston perustiedot
    print(f"Aineiston koko: {df.shape[0]} riviä, {df.shape[1]} saraketta")

    # Esikäsitellään tekstit
    df["processed_review"] = df["Review"].apply(preprocess_text)

    # Suoritetaan tunneanalyysi
    df["sentiment"] = df["Review"].apply(analyze_sentiment)

    # Eritellään avainsanat
    keywords = extract_keywords(df["processed_review"].tolist())

    # Vektorisoidaan tekstit
    tfidf_matrix, vectorizer = vectorize_texts(df["processed_review"].tolist())

    # Klusteroidaan arvostelut
    cluster_labels, kmeans_model = cluster_reviews(tfidf_matrix)

    # Haetaan klustereiden avainsanat
    cluster_keywords = get_cluster_keywords(vectorizer, kmeans_model)

    # Luodaan visualisoinnit
    create_visualizations(df, cluster_labels, keywords)

    # Tulostetaan yhteenveto
    print_summary(df, cluster_labels, keywords, cluster_keywords)

if __name__ == "__main__":
    main()