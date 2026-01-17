"""
Hallitsee vektoriupotusten luomisen tekstille käyttäen esikoulutettua mallia.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

class Embedder:
    """
    Luo upotuksia tekstilla käyttäen esikoulutettua transformers-mallia.
    Käyttää sentence-transformers yhteensopivaa mallia.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Parametrit ja attribuutit:
            model_name (str): Mallin nimi, oletuksena all_MiniLM-L6-v2.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Ladataan mallia {model_name} laitteelle {self.device}...")

        # Ladataan tokenisoija ja malli
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print("Malli ladattu onnistuneesti")

    def embed_text(self, text):
        """
        Luo upotukset tekstille tai listalle tekstejä.

        Parametrit:
            text (Union[str, List[str]]): Teksti tai lista tekstejä, joille upotukset luodaan.

        Palauttaa:
            np.ndarray: NumPy-taulukko upotuksista (muoto: [tekstien_lkm, upotusulottuvuus]).
        """
        # Muunnettaan yksittäinen teksti listaksi
        if isinstance(text, str):
            text = [text]

        # Tokenisoidaan tekstit
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Siirretään tensori laitteelle
        encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}

        # Luodaan tuptukset
        with torch.no_grad():
            model_output = self.model(**encoded_input)

            # Käytetään keskiarvopoolausta (mean pooling) tokenien upotuksista
            embeddings = self._mean_pooling(
                model_output,
                encoded_input["attention_mask"]
            )

            # Normalisoidaan upotukset
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Muunnetaan NumPy-taulukoksi
        embeddings_np = embeddings.cpu().numpy()

        return embeddings_np
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Suorittaa keskiarvopoolaukset tokenien upotuksista huomioiden huomiointimaskin.
        
        Parametrit:
            model_output: Mallin ulostulo.
            attention_mask: Huomiointimaski.

        Palauttaa:
            torch.Tensor: Poolatut upotukset.
        """
        # Haetaan tokenien upotukset
        token_embeddings = model_output[0]

        # Laajennetaan huomiointimaski ulottuvuuksien mukaan
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Lasketaan painotettu summa tokenien upotuksista
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # Lasketaan huomioitujen tokenien määrä
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Lasketaan keskiarvo
        return sum_embeddings / sum_mask
    
    def embed_batch(self, texts, batch_size=32):
        """
        Luo upotukset suurelle listalle tekstejä erissä muistitehokkaampi isommille aineistoille.)
        
        Parametrit:
            texts (List[str]): Lista tekstejä.
            batch_size (int): Erän koko, oletuksena 32.

        Palauttaa:
            np.ndarray: NumPy-taulukko upotuksista (muoto: [len(texts), upotusulottuvuus]).
        """
        all_embeddings = []

        # Käsitellään tekstit erissä
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_text(batch)
            all_embeddings.append(batch_embeddings)

        # Yhdistetään kaikki erien upotukset
        return np.vstack(all_embeddings)
    
    def get_embedding_dimension(self):
        """
        Hakee (tämän mallin) upotusten ulottuvuuden

        Palauttaa:
            int: Upotusten ulottuvuus.
        """
        # Luodaan testiupotus ulottuvuuden määrittämiseksi
        dummy_embedding = self.embed_text("test")
        return dummy_embedding.shape[1]
       
class EmbeddingCache:
    """Välimuisti dokumenttien upotusten tallentamiseen ja hakemiseen."""

    def __init__(self, cache_dir="data/embeddings"):
        """
        Parametrit/attribuutit:
            cache_dir (str): Hakemisto, johon upotukset tallennetaan.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, doc_id):
        """
        Luo dokumentille välimuistipolun.
        
        Parametrit:
            doc_id (str): Dokumentin tunniste.
            
        Palauttaa:
            str: Välimuistipolku.
        """
        return os.path.join(self.cache_dir, f"{doc_id}_embeddings.npy")
    
    def save_embeddings(self, doc_id, embeddings):
        """
        Tallentaa dokumentin upotukset välimuistiin.
        
        Parametrit:
            doc_id (str): Dokumentin tunniste.
            embeddings (np.ndarray): Tallennettavat upotukset.
        """
        cache_path = self._get_cache_path(doc_id)
        np.save(cache_path, embeddings)

    def _load_embeddings(self, doc_id):
        """
        Lataa upotukset välimuistista.
        
        Parametrit:
            doc_id (str): Dokumentin tunniste.
            
        Palauttaa:
            np.ndarray: Ladatut upotukset.
        """
        cache_path = self._get_cache_path(doc_id)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Upotuksia ei löytynyt välimuistista dokumentille {doc_id}")
        return np.load(cache_path)
    
    def delete_embeddings(self, doc_id):
        """
        Poistaa dokumentin upotukset välimuistista.
        
        Parametrit:
            doc_id (str): Dokumentin tunniste.
        """
        cache_path = self._get_cache_path(doc_id)
        if os.path.exists(cache_path):
            os.remove(cache_path)