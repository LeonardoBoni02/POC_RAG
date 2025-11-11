from datasets import load_dataset, Dataset
import os
import pandas as pd

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

            # --- IL FILTRO CORRETTO ---
            print("Filtraggio per dominio: 'dmv' (sulla colonna 'document')...")
            self.dmv_data = test_data.filter(
                lambda riga: riga["document"] == "dmv"
            )
            # --- FINE FILTRO ---

            print(f"FILTRO COMPLETATO! Dati 'dmv' trovati: {len(self.dmv_data)}.")
            return True

        except Exception as e:
            print(f"ERRORE CRITICO durante il caricamento/filtro: {e}")
            return False

    def save_dmv_data_to_csv(self, output_filename="dmv_data_filtrato.csv"):
        """Salva i dati filtrati in un file CSV usando Pandas."""
        if self.dmv_data and len(self.dmv_data) > 0:
            print(f"Salvataggio dei {len(self.dmv_data)} esempi 'dmv' in {output_filename}...")
            
            # --- ECCO LA MODIFICA ---
            # 1. Converti il dataset filtrato in un DataFrame Pandas
            df = self.dmv_data.to_pandas()
            
            # 2. Salva il DataFrame in CSV (questo non sbaglia)
            df.to_csv(output_filename, index=False)
            # --- FINE MODIFICA ---
            
            print(f"SALVATAGGIO COMPLETATO. Controlla il file {output_filename}.")
        else:
            print("Nessun dato 'dmv' da salvare.")

# --- Esegui questo script ---
if __name__ == "__main__":
    handler = DataHandler()
    
    # 1. Esegui il caricamento e filtro
    success = handler.load_and_filter_dmv_data()
    
    if success:
        # 2. SALVA IL CSV (con Pandas)
        handler.save_dmv_data_to_csv()