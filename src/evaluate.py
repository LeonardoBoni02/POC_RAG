# src/evaluation/evaluate.py

import os
import ast
import pandas as pd
import sys # <-- Importa SYS

# --- CORREZIONE IMPORT ---
# Aggiungi la cartella 'src' (che è /app/src) al PYTHONPATH
# per permettere gli import di 'retrieval' e 'generation'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- FINE CORREZIONE ---

from retrieval import Retriever
from generation import Generator

# === CONFIGURAZIONE ===
# Corretto: punta al file CSV nel volume 'data'
TEST_SET_PATH = "data/dmv_data_filtrato.csv" 
# Corretto: punta alla cartella di questo script
REPORT_PATH = "src/evaluation/evaluation_report.csv" 
MAX_SAMPLES = 50

# ========== FUNZIONI DI SUPPORTO: NORMALIZZAZIONE & METRICHE ==========
# (Il tuo codice qui è perfetto, lo ometto per brevità)
def _norm(text: str) -> str:
    text = str(text).lower().strip()
    return " ".join(text.split())

def exact_match(pred: str, gold: str) -> float:
    if not gold:
        return 0.0
    return 1.0 if _norm(pred) == _norm(gold) else 0.0

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = _norm(pred).split()
    gold_tokens = _norm(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    common = 0
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

# ========== PARSING SEMPLICE DI MESSAGES E ANSWERS ==========
# (Il tuo codice qui è perfetto, lo ometto per brevità)
def parse_answer(raw: str) -> str:
    try:
        data = ast.literal_eval(raw)
        if isinstance(data, list) and data:
            return str(data[0])
        return str(data)
    except Exception:
        return str(raw)

def parse_question(raw: str) -> str:
    try:
        data = ast.literal_eval(raw)
        if isinstance(data, list):
            user_msgs = [
                m.get("content", "")
                for m in data
                if isinstance(m, dict) and m.get("role") == "user"
            ]
            if user_msgs:
                return user_msgs[-1]
        return str(raw)
    except Exception:
        return str(raw)

# ========== PIPELINE DI VALUTAZIONE ==========

def run_evaluation():
    print("🚀 Avvio valutazione RAG (semplice)")

    # 1. Controllo esistenza test set
    if not os.path.exists(TEST_SET_PATH):
        print(f"❌ Test set non trovato: {TEST_SET_PATH}")
        print("Il file CSV non è ancora stato generato. Avvia 'docker-compose up' e attendi.")
        return

    # 2. Carico il CSV
    df = pd.read_csv(TEST_SET_PATH)
    df = df.dropna(subset=["messages", "answers"])
    df = df.head(MAX_SAMPLES)
    print(f"📄 Campioni di test utilizzati: {len(df)}")

    # 3. Inizializzo Retriever
    try:
        retriever = Retriever()
        # Il retriever caricherà l'indice da 'data/processed' che è
        # stato creato dal 'server.py' all'avvio.
        retriever.load_index() # Assicurati di caricarlo
    except Exception as e:
        print(f"❌ Errore inizializzazione Retriever: {e}")
        print("Assicurati che l'indice FAISS esista (generato da server.py all'avvio).")
        return

    # 4. Inizializzo Generator
    try:
        generator = Generator()
    except Exception as e:
        print(f"❌ Errore inizializzazione Generator: {e}")
        print("Controlla GOOGLE_API_KEY nel file .env.")
        return

    rows = []
    total_f1 = 0.0
    total_em = 0.0

    # 5. Loop sugli esempi
    for idx, (_, row) in enumerate(df.iterrows()):
        raw_messages = str(row["messages"])
        raw_answers = str(row["answers"])

        question = parse_question(raw_messages)
        gold = parse_answer(raw_answers)

        try:
            contexts = retriever.search(question, k=3)
            pred = generator.generate_answer(question, contexts)
        except Exception as e:
            pred = f"ERRORE: {e}"
            contexts = []

        f1 = f1_score(pred, gold)
        em = exact_match(pred, gold)
        total_f1 += f1
        total_em += em
        rows.append({
            "question": question,
            "ground_truth_answer": gold,
            "generated_answer": pred,
            "f1": f1,
            "em": em,
            "contexts": " ||| ".join(contexts),
        })

        if idx < 3:
            print(f"\n--- ESEMPIO DEBUG {idx} ---")
            print("Q:", question)
            print("PRED:", pred)
            print("F1:", f1)

    # 6. Salvo il report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_PATH, index=False)

    # 7. Stampo le metriche
    n = len(rows)
    avg_f1 = total_f1 / n if n else 0.0
    avg_em = total_em / n if n else 0.0

    print("\n✅ Valutazione conclusa.")
    print(f"📁 Report salvato in: {REPORT_PATH}")
    print(f"📊 F1 medio: {avg_f1:.4f}")
    print(f"📊 EM medio: {avg_em:.4f}")

if __name__ == "__main__":
    run_evaluation()