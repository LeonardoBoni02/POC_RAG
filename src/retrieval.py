from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json


class Retriever:
    """
    Gestisce la creazione di embedding e la ricerca
    nel database vettoriale (FAISS).
    """

    def __init__(
        self,
        model_name= "all-MiniLM-L6-v2",
        store_path= "data/processed/vector_store",
        index_name= "dmv.index",
        chunks_name= "dmv_chunks.json",
    ):
        """
        Inizializza il modello di embedding e i path di salvataggio.
        """
        self.model = SentenceTransformer(model_name)
        self.store_path = store_path
        self.index_path = os.path.join(store_path, index_name)
        self.chunks_path = os.path.join(store_path, chunks_name)

        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        """
        Crea l'indice FAISS a partire dagli embeddings dei chunks
        e salva sia l'indice che i chunks su disco.
        """
        print("✅ Creazione embeddings...")
        embeddings = self.model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        d = embeddings.shape[1]  # dimensione vettori
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        self.chunks = chunks

        # Creiamo la cartella se non esiste
        os.makedirs(self.store_path, exist_ok=True)

        # Salviamo l'indice
        faiss.write_index(self.index, self.index_path)

        # Salviamo i chunk (in modo semplice)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ Indice costruito e salvato in: {self.index_path}")
        print(f"✅ Chunks salvati in: {self.chunks_path}")

    def load_index(self):
        """
        Carica indice FAISS e chunks da disco.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
            raise FileNotFoundError(
                "Indice o file dei chunks non trovato. Costruisci prima l'indice con build_index()."
            )

        self.index = faiss.read_index(self.index_path)

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("✅ Indice e chunks caricati da disco.")

    def search(self, query: str, k = 3):
        """
        Ritorna i k chunks più rilevanti per la query.
        """
        # Se l'indice non è in memoria, proviamo a caricarlo
        if self.index is None or not self.chunks:
            self.load_index()

        # Evitiamo di chiedere più risultati di quelli disponibili
        k = min(k, len(self.chunks))

        # Embedding della query
        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
        ).astype("float32")

        distances, indices = self.index.search(query_vec, k)

        # indices[0] contiene gli indici dei migliori chunk
        return [self.chunks[i] for i in indices[0]]

'''
# --- Esegui questo script ---
if __name__ == "__main__":
    retriever = Retriever()

    # Esempio didattico di chunk testuali
    chunks = [
        "The DMV requires every motorist to maintain auto insurance.",
        "You can renew your driver's license online or in person at a local branch.",
        "Make sure to update your address if you move to a new residence.",
        "Traffic violations can result in fines or license suspension."
    ]

    # 1️⃣ Costruisci l'indice FAISS
    retriever.build_index(chunks)

    # 2️⃣ Fai una query di prova
    query = "How do I update my address?"
    risultati = retriever.search(query, k=2)

    print("\n🔎 Query:", query)
    print("Risultati trovati:")
    for i, testo in enumerate(risultati, start=1):
        print(f"\n--- Chunk {i} ---")
        print(testo)
'''