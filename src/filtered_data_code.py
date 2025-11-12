from datasets import load_dataset, Dataset
import os
import pandas as pd

# Definisci il percorso dati relativo
DATA_DIR = "data"
CSV_NAME = "dmv_data_filtrato.csv"
OUTPUT_PATH = os.path.join(DATA_DIR, CSV_NAME)

class DataHandler:
    """
    Carica lo split 'test', lo filtra per 'dmv' (sulla colonna 'document')
    e salva il risultato usando Pandas.
    """
    def __init__(self, dataset_name="nvidia/ChatRAG-Bench", subset="doc2dial"):
        self.dataset_name = dataset_name
        self.subset = subset
        self.dmv_data = None  
        print(f"DataHandler pronto per {self.dataset_name}/{self.subset}.")

    def load_and_filter_dmv_data(self):
        """
        Carica lo split 'test' e lo filtra per 'dmv'.
        """
        print(f"Caricamento split 'test' da {self.dataset_name}...")
        try:
            dataset_dict = load_dataset(self.dataset_name, self.subset)
            test_data = dataset_dict['test']
            print(f"Dati 'test' originali caricati: {len(test_data)} esempi.")

            print("Filtraggio per dominio: 'dmv' (sulla colonna 'document')...")
            self.dmv_data = test_data.filter(
                lambda riga: riga["document"] == "dmv"
            )
            print(f"FILTRO COMPLETATO! Dati 'dmv' trovati: {len(self.dmv_data)}.")
            return True

        except Exception as e:
            print(f"ERRORE CRITICO durante il caricamento/filtro: {e}")
            return False

    def save_dmv_data_to_csv(self, output_filename=OUTPUT_PATH): # Usa la variabile
        """Salva i dati filtrati in un file CSV usando Pandas."""
        if self.dmv_data and len(self.dmv_data) > 0:
            print(f"Salvataggio dei {len(self.dmv_data)} esempi 'dmv' in {output_filename}...")
            
            # Assicurati che la cartella esista
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            
            df = self.dmv_data.to_pandas()
            df.to_csv(output_filename, index=False)
            
            print(f"SALVATAGGIO COMPLETATO. Controlla il file {output_filename}.")
        else:
            print("Nessun dato 'dmv' da salvare.")

# --- Esegui questo script ---
if __name__ == "__main__":
    # Questo script verrà eseguito all'avvio del container
    # per assicurarsi che i dati CSV esistano.
    if os.path.exists(OUTPUT_PATH):
        print(f"Il file {OUTPUT_PATH} esiste già. Salto il download.")
    else:
        print(f"Il file {OUTPUT_PATH} non esiste. Avvio download e filtro...")
        handler = DataHandler()
        success = handler.load_and_filter_dmv_data()
        
        if success:
            handler.save_dmv_data_to_csv()