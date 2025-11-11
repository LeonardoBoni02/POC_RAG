# src/data_processing.py
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    '''
    Classe per caricare e processare i documenti
    per la nostra knowledge base.
    '''
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        '''
        Inizializza il text splitter.
        '''
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_documents_from_csv(self, csv_path = "C:\\Users\\Daniele\\Desktop\\POC\\POC_RAG\\data\\dmv_data_filtrato.csv"):
        
        try:
            csv.field_size_limit(50_000_000)
        except OverflowError:
            # Aumentato limite visualizzazione csv tramite terminale, se non funzionante, si ripiega su un valore più basso
            csv.field_size_limit(10_000_000)

        documents: list[str] = []
        visti: set[str] = set()  # per tenere traccia dei testi già aggiunti

        # Apriamo il CSV in modalità lettura
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Per ogni riga del CSV costruiamo un testo
            for row in reader:
                # Prendiamo i vari pezzi (se vuoti li ignoriamo)
                parti_testo: list[str] = []

                # Testo principale del documento
                if row.get("document"):
                    parti_testo.append(row["document"].strip())

                # Contesto "vero" usato come riferimento
                if row.get("ground_truth_ctx"):
                    parti_testo.append("Ground truth context:\n" + row["ground_truth_ctx"].strip())

                # Contesti recuperati (es. da motore di ricerca)
                if row.get("ctxs"):
                    parti_testo.append("Retrieved contexts:\n" + row["ctxs"].strip())

                # Conversazione utente-modello
                if row.get("messages"):
                    parti_testo.append("Messages:\n" + row["messages"].strip())

                # Risposta finale
                if row.get("answers"):
                    parti_testo.append("Answer:\n" + row["answers"].strip())

                # Uniamo tutte le parti in un unico stringone
                testo_documento = "\n\n".join(parti_testo)

                # Se è vuoto, lo saltiamo
                if not testo_documento:
                    continue

                # Deduplica semplice: aggiungiamo solo se non l'abbiamo già visto
                if testo_documento not in visti:
                    visti.add(testo_documento)
                    documents.append(testo_documento)

        return documents

    def split_documents(self, documents_text):
        """
        Divide i testi dei documenti in chunks.
        """
        chunks = self.text_splitter.create_documents(documents_text)
        # Estrae il contenuto di testo (page_content)
        return [chunk.page_content for chunk in chunks]
    
'''
# --- Esegui questo script ---

if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.load_documents_from_csv()
    print(f" Documenti caricati: {len(docs)}")

    chunks = processor.split_documents(docs)
    print(f" Chunk generati: {len(chunks)}")

    # opzionale: vedi un esempio
    if chunks:
        print("\n=== Primo chunk ===")
        print(chunks[0][:500])

docs = processor.load_documents_from_csv()
print(docs[1])  # Stampa il secondo documento

'''