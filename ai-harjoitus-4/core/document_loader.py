"""
Hallitsee dokumenttien metatiedot, niiden lataamisen ja pilkkoo ne hallittaviin osiin. 
"""

from datetime import datetime
from typing import List
import os
import pypdf
import re

class DocumentChunk:
    """Kuvaa dokumentin osaa ja metatietoja tekstidokumentissa."""

    def __init__(self, text, doc_id, chunk_id, metadata):
        """
        Parametrit ja attribuutit:
            text (str): Osan teksti.
            doc_id (str): Dokumentin tunniste.
            chunk_id (int): Osan tunniste dokumentissa.
            metadata (dict): Dokumentin metatiedot.
        """
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.metadata = metadata or {}

    def _to_dict(self):
        """
        Muuntaa osan sanakirjaksi.

        Palauttaa:
            dict: Sanakirja, joka sisältää osan tiedot.
        """
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def _from_dict(cls, data):
        """
        Luo DocumentChunk-olion sanakirjasta.

        Parametrit:
            data (dict): Sanakirja, joka sisältää osan tiedot.

        Palauttaa:
            DocumentChunk: Instanssi.
        """
        return cls(
            text=data["text"],
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            metadata=data.get("metadata", {})
        )
    
class Document:
    """Kuvaa tekstidokumenttia, osia ja sen metatietoja."""

    def __init__(self, doc_id, filename, filepath, content, file_size):
        """        
        Parametrit ja attribuutit:
            doc_id (str): Dokumentin tunniste.
            filename (str): Tiedoston nimi.
            filepath (str): Tiedoston polku.
            content (str): Dokumentin koko teksti.
            file_size (int): Tiedoston koko tavuina.
        """
        self.doc_id = doc_id
        self.filename = filename
        self.filepath = filepath
        self.content = content
        self.file_size = file_size
        self.upload_date = datetime.now().isoformat()
        self.chunks: List[DocumentChunk] = []

    def _to_dict(self):
        """
        Muuntaa dokumentin sanakirjaksi.

        Palauttaa:
            dict: Sanakirja, joka sisältää dokumentin tiedot.
        """
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "filepath": self.filepath,
            "content": self.content,
            "file_size": self.file_size,
            "upload_date": self.upload_date,
            "chunks": [chunk._to_dict() for chunk in self.chunks]
        }
    
    @classmethod
    def _from_dict(cls, data):
        """
        Luo Document-olion sanakirjasta.

        Parametrit:
            data (dict): Sanakirja, joka sisältää dokumentin tiedot.

        Palauttaa:
            Document: Instanssi.
        """
        doc = cls(
            doc_id=data["doc_id"],
            filename=data["filename"],
            filepath=data["filepath"],
            content=data["content"],
            file_size=data["file_size"]
        )
        doc.upload_date = data["upload_date"]
        doc.chunks = [DocumentChunk._from_dict(c) for c in data.get("chunks", [])]
        return doc
    
class DocumentLoader:
    """Käsittelee dokumenttien lataamisen ja käsittelyn."""

    def __init__(self, chunk_size=512, chunk_overlap=50):
        """
        Parametrit:
            chunk_size (int): Osan maksimipituus merkeissä.
            chunk_overlap (int): Merkkien määrä, joka ylittää osien välillä.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, filepath):
        """
        Lataa dokumentin sisällön tiedostosta (.txt tai .pdf).

        Parametrit:
            filepath (str): Tiedoston polku.

        Palauttaa:
            tuple: (dokumentin sisältö (str), tiedoston koko tavuina (int)).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tiedostoa {filepath} ei löytynyt")
        
        file_size = os.path.getsize(filepath)
        extension = os.path.splitext(filepath)[1].lower()

        if extension == ".txt":
            content = self._load_txt(filepath)
        elif extension == ".pdf":
            content = self._load_pdf(filepath)
        else:
            raise ValueError(f"Tiedostotyyppiä {extension} ei tueta")
        
        return content, file_size
        
    def _load_txt(self, filepath):
        """
        Lataa sisällön .txt-tiedostosta.

        Parametrit:
            filepath (str): Tiedoston polku.

        Palauttaa:
            str: Tiedoston sisältö.
        """
        with open(filepath, 'r', encoding="utf-8") as f:
            return f.read()
        
    def _load_pdf(self, filepath):
        """
        Lataa sisällön .pdf-tiedostosta.

        Parametrit:
            filepath (str): Tiedoston polku.

        Palauttaa:
            str: Tiedoston sisältö.
        """
        text = []
        try:
            with open(filepath, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as e:
            raise ValueError(f"Virhe PDF-tiedoston lataamisessa: {e}")

        return "\n".join(text)
    
    def create_document(self, filepath, doc_id=None):
        """
        Luo Document-olion tiedostosta.

        Parametrit:
            filepath (str): Tiedoston polku.
            doc_id (str | None): Dokumentin tunniste.

        Palauttaa:
            Document: Olio, jossa dokumentin sisältö.
        """
        filename = os.path.basename(filepath)
        content, file_size = self.load_file(filepath)

        if doc_id is None:
            # Luodaan uniikki ID tiedostonimen ja ajan perusteella
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            doc_id = f"{os.path.splitext(filename)[0]}_{timestamp}"

        document = Document(
            doc_id=doc_id,
            filename=filename,
            filepath=filepath,
            content=content,
            file_size=file_size
        )

        # Luodaan osat
        document.chunks = self.chunk_text(content, doc_id)

        return document
    
    def chunk_text(self, text, doc_id):
        """
        Pilkkoo tekstin päällekkäisiin osiin.

        Parametrit:
            text (str): Teksti, joka pilkotaan.
            doc_id (str): Dokumentin tunniste.

        Palauttaa:
            list: Lista DocumentChunk-olioita.
        """
        # Siistitään ja normalisoidaan teksti
        # Poistetaan ylimääräiset välilyönnit ja rivinvaihdot
        text = re.sub(r"\s+", ' ', text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # Lasketaan loppupaikka
            end = start + self.chunk_size

            # Jos tämä ei ole viimeinen osa, yritetään katkaista lauseeseen tai sanaan.
            if end < len(text):
                # Etsitään lauseen lopetusmerkkiä
                sentence_end = self._find_sentence_boundary(text, end)
                if sentence_end > start:
                    end = sentence_end
                else:
                    # Etsitään viimeinen välilyönti (sanan loppu)
                    for i in range(end - 1, start, -1):
                        if text[i].isspace():
                            word_end = i + 1
                    word_end = end

                    if word_end > start:
                        end = word_end

            # Erotellaan osan teksti
            chunk_text = text[start:end].strip()

            if chunk_text: # Lisätään vain osat, joissa on tekstiä
                chunk = DocumentChunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    metadata={"start_position": start, "end_position": end}
                )
                chunks.append(chunk)
                chunk_id += 1

            # Päivitetään aloituspaikka seuraavaa osaa varten
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _find_sentence_boundary(self, text, end):
        """
        Etsii lähimmän lauseen lopetusmerkin ennen osan loppua.

        Parametrit:
            text (str): Koko teksti.
            end (int): Alueen loppuindeksi.

        Palauttaa:
            int: Lopetusmerkin sijainti tai -1, jos ei löydy.
        """
        # Etsitään lauseen lopetusmerkkejä viimeisestä 20%:sta tätä osaa
        search_start = end - (self.chunk_size // 5)
        search_text = text[search_start:end]

        # Etsitään viimeisin lauseen lopetusmerkki
        for delimiter in [". ", ", ", "! ", "? ", ".\n", "!\n", "?\n"]:
            position = search_text.rfind(delimiter)
            if position != -1:
                return search_start + position + len(delimiter)
            
        return -1
    
    def update_chunk_size(self, chunk_size, chunk_overlap):
        """
        Päivittää osan koon.

        Parametrit:
            chunk_size (int): Uusi osan koko.
            chunk_overlap (int): Uusi osien päällekkäisyys.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap