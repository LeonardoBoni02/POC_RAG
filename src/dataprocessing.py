# src/data_processing.py
import csv
import os # Importa OS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Definisci il percorso dati relativo
DATA_DIR = "data"
CSV_NAME = "dmv_data_filtrato.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)

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

    def load_documents_from_csv(self, csv_path=CSV_PATH): # Usa la variabile
        
        # Assicurati che la cartella dati esista prima di leggere
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if not os.path.exists(csv_path):
             print(f"Attenzione: file {csv_path} non ancora esistente. Verrà creato dallo script di setup.")
             return [] # Ritorna lista vuota se il file non c'è

        try:
            csv.field_size_limit(50_000_000)
        except OverflowError:
            csv.field_size_limit(10_000_000)

        documents: list[str] = []
        visti: set[str] = set()

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                parti_testo: list[str] = []
                if row.get("document"):
                    parti_testo.append(row["document"].strip())
                if row.get("ground_truth_ctx"):
                    parti_testo.append("Ground truth context:\n" + row["ground_truth_ctx"].strip())
                if row.get("ctxs"):
                    parti_testo.append("Retrieved contexts:\n" + row["ctxs"].strip())
                if row.get("messages"):
                    parti_testo.append("Messages:\n" + row["messages"].strip())
                if row.get("answers"):
                    parti_testo.append("Answer:\n" + row["answers"].strip())

                testo_documento = "\n\n".join(parti_testo)
                if not testo_documento:
                    continue
                if testo_documento not in visti:
                    visti.add(testo_documento)
                    documents.append(testo_documento)

        return documents

    def split_documents(self, documents_text):
        """
        Divide i testi dei documenti in chunks.
        """
        chunks = self.text_splitter.create_documents(documents_text)
        return [chunk.page_content for chunk in chunks]

'''
# Esecuzione non necessaria qui, gestita dal server
'''