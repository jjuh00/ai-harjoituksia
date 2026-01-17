"""
Tarjoaa RAG-järjestelmää tukevia apufunktioita, kuten
tiedostotoimitoja, tekstinkäsittelyä ja konfiguraationhallintaa.
"""

import os
import json
import numpy as np

def ensure_directory_exists(directory_path):
    """
    Varmistaa, että hakimisto on olemassa ja luo sen tarvittaessa.

    Parametrit:
        directory_path (str): Hakemiston polku.
    """
    os.makedirs(directory_path, exist_ok=True)

def save_json(data, filepath):
    """
    Tallentaa tiedot JSON-tiedostoon.

    Parametrit:
        data (Any): Tallennettavat tiedot (täytyy olla JSON-yhteensopivia).
        filepath (str): Tiedoston polku.
    """
    with open(filepath, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath):
    """
    Lataa tiedot JSON-tiedostosta.

    Parametrit:
        filepath (str): Tiedoston polku.

    Palauttaa:
        Any: Ladatut tiedot.
    """    
    with open(filepath, 'r', encoding="utf-8") as f:
        return json.load(f)
    
def truncate_text(text, max_length = 100):
    """
    Katkaisee tekstin enimmäispituuteen, tarvittaessa lisää "..." loppuun.

    Parametrit:
        text (str): Teksti, joka katkaistaan.
        max_length (int): Enimmäispituus merkkeinä.

    Palauttaa:
        str: Katkaistu teksti.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def cosine_similarity(vec1, vec2):
    """
    Laskee kosinietäisyyden kahden vektorin välillä.

    Parametrit:
        vec1 (np.ndarray): Ensimmäinen vektori.
        vec2 (np.ndarray): Toinen vektori.

    Palauttaa:
        float: Kosinietäisyys vektorien välillä.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float((dot_product) / (norm1 * norm2))

def format_document_size(size_in_bytes):
    """
    Muuntaa tiedostokoon luettavampaan muotoon.

    Parametrit:
        size_in_bytes (int): Tiedostokoko tavuina.

    Palauttaa:
        str: Muunnettu tiedostokoko (esim. "2.5 MB").
    """
    for unit in ['B', "KB", "MB", "GB"]:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} TB"

class Config:
    """
    RAG-järjestelmän konfiguraatioasetukset.
    """

    def __init__(self):
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.top_k_chunks = 3
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.llm_model_name = "google/flan-t5-base"
        self.max_context_length = 2048
        self.data_directory = "data"

    def _to_dict(self):
        """
        Muuntaa konfiguraation sanakirjaksi.

        Palauttaa:
            dict: Sanakirja, joka sisältää konfiguraation tiedot.
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_chunks": self.top_k_chunks,
            "model_name": self.model_name,
            "llm_model_name": self.llm_model_name,
            "max_context_length": self.max_context_length,
            "data_directory": self.data_directory
        }
    
    def _from_dict(self, config_dict):
        """
        Lataa konfiguraation sanakirjasta.

        Parametrit:
            config_dict (dict): Sanakirja, joka sisältää konfiguraation tiedot.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, filepath):
        """
        Tallentaa konfiguraation tiedostoon.

        Parametrit:
            filepath (str): Tiedoston polku.
        """
        save_json(self._to_dict(), filepath)

    def load(self, filepath):
        """
        Lataa konfiguraation tiedostosta.

        Parametrit:
            filepath (str): Tiedoston polku.
        """
        if os.path.exists(filepath):
            config_dict = load_json(filepath)
            self._from_dict(config_dict)