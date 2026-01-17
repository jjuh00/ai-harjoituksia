"""
Käsittelee toiminnot LLM:n kanssa vastausten luomiseksi kontekstin
ja käyttäjän syötteen perusteella.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMClient:
    """
    Toiminnot LLM:n kanssa vastausten luomiseksi. Käyttää siihen transformers-kirjastoa.
    """

    def __init__(self, model_name="google/flan-t5-base", max_length=512):
        """
        Parametrit ja attribuutit:
            model_name (str): LLM-mallin nimi.
            max_length (int): Maksimipituus vastaukselle.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Ladataan mallia {model_name} laitteelle {self.device}...")

        # Ladataan tokenisoija ja malli
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        print("Malli ladattu onnistuneesti")

    def generate_answer(self, query, context_chunks, max_context_length):
        """
        Luo vastauksen kyselyyn haetun kontekstin perusteella.

        Parametrit:
            query (str): Käyttäjän kysymys.
            context_chunks (list[RetrievalResult]): Lista haetuista kontekstikappaleista (RetrievalResult-olioita).
            max_context_length (int): Maksimipituus kontekstille tokenoituina.

        Palauttaa:
            dict: Sanakirja, jossa on luotu vastaus, lähteet ja mallin käyttämä konteksti.
        """
        if not context_chunks:
            return {
                "answer": "Unfortunately, I couldn't find information related to your question in the documents",
                "sources": [],
                "used_context": ""
            }
        
        # Rakennetaan konteksti yhdistämällä haetut dokumentin osat
        context, sources = self._build_context(context_chunks, max_context_length)

        # Luodaan kehote mallille
        prompt = f"""Based on the following context from the documents, please answer the question.
                If the answer cannot be found in the context, say "I cannot answer this question based on the provided documents."
                Context: {context}
                Question: {query}
                Answer:"""
        
        # Luodaan vastaus
        answer = self._generate_text(prompt)

        return {
            "answer": answer,
            "sources": sources,
            "used_context": context
        }
    
    def _build_context(self, chunks, max_length):
        """
        Rakentaa kontekstin yhdistämällä haetut dokumentin osat.

        Parametrit:
            chunks (list[RetrievalResult]): Lista haetuista kontekstikappaleista (RetrievalResult-olioita).
            max_length (int): Maksimipituus kontekstille tokenoituina.

        Palauttaa:
            tuple (str, list): Yhdistetty konteksti merkkijonona ja lähteiden lista.
        """
        context_parts = []
        sources = []
        current_length = 0

        for i, result in enumerate(chunks):
            chunk_text = result.chunk.text
            chunk_length = len(chunk_text)

            # Tarkistetaan, ylittäisikö tämän osan lisääminen maksimipituuden
            if current_length + chunk_length > max_length and context_parts:
                break

            # Lisätään osat kontekstiin
            context_parts.append(f"[Document {i+1}]: {chunk_text}")
            current_length += chunk_length

            # Lisätään lähdetiedot
            sources.append({
                "chunk_id": i + 1,
                "doc_filename": result.doc_filename,
                "doc_id": result.chunk.doc_id,
                "similarity_score": float(result.score),
                "text_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            })

        context_string = "\n\n".join(context_parts)
        return context_string, sources

    def _generate_text(self, prompt):
        """
        Luo tekstin käyttämällä kielimallia.

        Parametrit:
            prompt (str): Kehote mallille.

        Palauttaa:
            str: Luotu teksti.
        """
        # Tokenisoidaan syöte
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Luodaan vastaus
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )

        # Dekoodataan ja palautetaan vastaus
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    def answer_question(self, query, context_chunks):
        """
        Yksinkertainen rajapinta vain kysymykseen vastauksen luomiseksi.

        Parametrit:
            query (str): Käyttäjän kysymys.
            context_chunks (list[RetrievalResult]): Lista haetuista kontekstikappaleista (RetrievalResult-olioita).

        Palauttaa:
            str: Luotu vastaus.
        """
        result = self.generate_answer(query, context_chunks, max_context_length=1000)
        return result["answer"]

    def get_model_info(self):
        """
        Hakee tiedot käytetystä mallista.

        Palauttaa:
            dict: Sanakirja mallin tiedoista.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "vocabulary_size": self.tokenizer.vocab_size
        }