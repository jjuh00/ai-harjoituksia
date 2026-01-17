"""
Suorittaa vektorihakuun liittyviä toimintoja.
"""

from core.embedder import Embedder
from typing import Dict, List, Tuple
from core.document_loader import Document, DocumentChunk
import numpy as np

class RetrievalResult:
    """Kuvaa haun tulosta sen samankaltaisuuspisteineen."""
    
    def __init__(self, chunk, score, doc_filename=''):
        """
        Parametrit ja attribuutit:
            chunk (DocumentChunk): Haettu dokumentin osa.
            score (float): Samankaltaisuuspisteet (0-1).
            doc_filename (str): Alkuperäisen dokumentin tiedostonimi.
        """
        self.chunk = chunk
        self.score = score
        self.doc_filename = doc_filename

    def __lt__(self, other):
        """Vertaa kahta hakutulosta niiden samankaltaisuuspisteiden perusteella."""
        return self.score < other.score
    
class Retriever:
    """
    Käsittelee samankaltaisuuteen perustuvaa hakua dokumenttien osista.
    Käyttää Embedder-luokkaa (kosinietäisyyttä) upotusten laskemiseen ja vertailuun.
    """

    def __init__(self, embedder: Embedder, top_k=3):
        """
        Parametrit ja attribuutit:
            embedder (Embedder): Upotusten laskemiseen käytettävä Embedder-olio.
            top_k (int): Haettavien osien määrä per kysely.
        """
        self.embedder = embedder
        self.top_k = top_k

        # Tallennustila dokuemnteille ja niiden upotuksille
        self.documents: Dict[str, Document] = {}
        self.chunk_embeddings: Dict[str, np.ndarray] = {} # doc_id -> upotukset
        self.all_chunks: List[Tuple[DocumentChunk, str]] = [] # (chuk, doc_id)
        self.all_embeddings: np.ndarray = None

    def add_document(self, document, embeddings):
        """
        Lisää dokumentin ja sen upotukset hakukokoelmaan.
        
        Parametrit:
            document (Document): Document-olio ja sen osat
            embeddings (np.ndarray): NumPy-taulukko, jossa dokumentin osien upotukset (muoto: [osien_lkm, upotusulottuvuus]).
        """
        if len(document.chunks) != embeddings.shape[0]:
            raise ValueError(f"Osien lukumäärä ({len(document.chunks)}) ei vastaa upotusten lukumäärää ({embeddings.shape[0]})")
        
        # Tallennetaan dokumentti ja sen upotukset
        self.documents[document.doc_id] = document
        self.chunk_embeddings[document.doc_id] = embeddings

        # Lisätään osat yleiseen listaan
        for chunk in document.chunks:
            self.all_chunks.append((chunk, document.filename))

        # Yhdistetään kaikki upotukset yhdeksi taulukoksi
        self._rebuild_embedding_matrix()

    def remove_document(self, doc_id):
        """
        Poistaa dokumentin ja sen osat hakukokoelmasta.

        Parametrit:
            doc_id (str): Poistettavan dokumentin tunniste.
        """
        if doc_id not in self.documents:
            return
        
        # Poistetaan tallennustilasta
        document = self.documents.pop(doc_id)
        self.chunk_embeddings.pop(doc_id)

        # Poistetaan osat yleisestä listasta
        self.all_chunks = [
            (chunk, filename) for chunk, filename in self.all_chunks
            if chunk.doc_id != doc_id
        ]

        # Yhdistetään kaikki upotukset uudelleen
        self._rebuild_embedding_matrix()

    def _rebuild_embedding_matrix(self):
        """Yhdistää kaikki osien upotukset yhdeksi NumPy-taulukoksi."""
        if not self.chunk_embeddings:
            self.all_embeddings = None
            return
        
        # Yhdistetään kaikki upotukset järjestyksessä
        embedding_list = []
        for chunk, _ in self.all_chunks:
            doc_embeddings = self.chunk_embeddings[chunk.doc_id]
            embedding_list.append(doc_embeddings[chunk.chunk_id])

        if embedding_list:
            self.all_embeddings = np.vstack(embedding_list)
        else:
            self.all_embeddings = None

    def retrieve(self, query, top_k=None):
        """
        Hakee relevantit dokumenttien osat kyselyn perusteella.

        Parametrit:
            query (str): Käyttäjän kysely
            top_k (int): Haettavien osien määrä

        Palauttaa:
            list[RetrievalResult]: Lista hakutuloksist, järjestettynä relevanssin mukaan (relevantein ensin).
        """
        if top_k is None:
            top_k = self.top_k

        if self.all_embeddings is None or self.all_embeddings.size == 0:
            return []
        
        # Luodaan kyselyn upotus
        query_embedding = self.embedder.embed_text(query)

        # Lasketaan kosinietäisyydet
        similarities = self._compute_similarities(query_embedding, self.all_embeddings)

        # Haetaan top_k indeksit käyttäen kasaa (heap) tehokkuuden parantamiseksi.
        top_k = min(top_k, len(similarities))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]

        # Järjestetään top_k indeksit pisteiden mukaan laskevasti
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Luodaan hakutulokset
        results = []
        for index in top_indices:
            chunk, doc_filename = self.all_chunks[index]
            score = similarities[index]
            result = RetrievalResult(chunk, score, doc_filename)
            results.append(result)

        return results
    
    def _compute_similarities(self, query_embedding, chunk_embeddings):
        """
        Laskee kosinietäisyydet kyselyn ja dokumenttien osien välillä.

        Parametrit:
            query_embedding (np.ndarray): Kyselyn upotusvektori (muoto: [1, upotusulottuvuus]).
            chunk_embeddings (np.ndarray): Dokumenttien osien upotukset (muoto: [osien_lkm, upotusulottuvuus]).

        Palauttaa:
            np.ndarray: Kosinietäisyydet (muoto: [osien_lkm]).
        """ 
        # Varmistetaan, että kyselyn upotus on 1-ulotteinne
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.squeeze()

        # Normalisoidaan upotukset
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        chunk_norms = chunk_embeddings / np.linalg.norm(
            chunk_embeddings, axis=1, keepdims=True
        )

        # Lasketaan kosinietäisyydet
        return np.dot(chunk_norms, query_norm)

    def get_document_count(self):
        """
        Palauttaa indeksoitujen dokumenttien lukumäärän.

        Palauttaa:
            int: Dokumenttien lukumäärä.
        """
        return len(self.documents)
    
    def _get_chunk_count(self):
        """
        Palauttaa indeksoitujen dokumenttien osien lukumäärän.

        Palauttaa:
            int: Dokumenttien osien lukumäärä.
        """
        return len(self.all_chunks)
    
    def clear(self):
        """Tyhjentää kaikki indeksoidut dokumentit ja upotukset."""
        self.documents.clear()
        self.chunk_embeddings.clear()
        self.all_chunks.clear()
        self.all_embeddings = None

    def update_top_k(self, top_k):
        """
        Päivittää haettavien osien määrän per kysely.

        Parametrit:
            top_k (int): Haettavien osien määrä.
        """
        self.top_k = max(1, top_k)

    def get_statistics(self):
        """
        Hakee tilastoja indeksoiduista dokumenteista.
        
        Palauttaa:
            dict: Sanakirja, joka sisältää tilastotiedot.
        """
        total_chunks = self._get_chunk_count()
        total_documents = self.get_document_count()

        average_chunks_per_document = total_chunks / total_documents if total_documents > 0 else 0

        return {
            "document_count": total_documents,
            "chunk_count": total_chunks,
            "average_chunks_per_document": average_chunks_per_document,
            "embedding_dimension": self.all_embeddings.shape[1] if self.all_embeddings is not None else 0
        }