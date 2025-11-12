# evaluation/evaluate.py

import os
import ast
import pandas as pd


from retrieval import Retriever
from generation import Generator

# === CONFIGURAZIONE ===
TEST_SET_PATH = "C:\\Users\\Daniele\\Desktop\\POC\\POC_RAG\\data\\dmv_data_filtrato.csv"
REPORT_PATH = "evaluation/evaluation_report.csv"
MAX_SAMPLES = 50  # per non sprecare troppe chiamate API: puoi aumentare se vuoi


# ========== FUNZIONI DI SUPPORTO: NORMALIZZAZIONE & METRICHE ==========

def _norm(text: str) -> str:
    """
    Normalizza il testo:
    - minuscole
    - spazi doppi rimossi
    Serve per rendere più "giusto" il confronto tra stringhe.
    """
    text = str(text).lower().strip()
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    """
    Exact Match molto semplice:
    1.0 se la risposta predetta (normalizzata) è identica alla gold,
    0.0 altrimenti.
    """
    if not gold:
        return 0.0
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    """
    F1 stile QA:
    misura quanto overlap c'è tra i token della risposta generata e quelli della risposta gold.
    È severa, ma utile per capire se almeno ci sono parole in comune rilevanti.
    """
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

def parse_answer(raw: str) -> str:
    """
    Nel dataset, 'answers' spesso è una lista in forma di stringa:
        "['testo della risposta']"
    Qui:
    - se è lista, prendiamo il primo elemento
    - altrimenti restituiamo la stringa così com'è.
    """
    try:
        data = ast.literal_eval(raw)
        if isinstance(data, list) and data:
            return str(data[0])
        return str(data)
    except Exception:
        return str(raw)


def parse_question(raw: str) -> str:
    """
    Nel dataset, 'messages' spesso è una lista di dizionari in forma di stringa:
        "[{'role': 'user', 'content': '...'}, {'role': 'assistant', ...}, ...]"

    Qui facciamo una cosa semplice e sensata:
    - se riusciamo a leggerla come lista:
        prendiamo l'ULTIMO messaggio con role='user'
    - se qualcosa va storto:
        usiamo la stringa così com'è.

    Questo ci dà una query pulita per il retrieval.
    """
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
        print("Verifica il percorso o il nome del file CSV.")
        return

    # 2. Carico il CSV e filtro righe utili
    df = pd.read_csv(TEST_SET_PATH)
    df = df.dropna(subset=["messages", "answers"])
    df = df.head(MAX_SAMPLES)  # prendiamo solo i primi N per il POC

    print(f"📄 Campioni di test utilizzati: {len(df)}")

    # 3. Inizializzo Retriever
    try:
        retriever = Retriever()
        # Il tuo Retriever dovrebbe caricare indice+chunks da disco (load_index interno in search/load_index).
        # Assicurati solo di aver costruito il vector store almeno una volta prima.
    except Exception as e:
        print(f"❌ Errore inizializzazione Retriever: {e}")
        return

    # 4. Inizializzo Generator (Gemini 2.0 Flash)
    try:
        generator = Generator()
    except Exception as e:
        print(f"❌ Errore inizializzazione Generator: {e}")
        print("Controlla GOOGLE_API_KEY o il file .env.")
        return

    rows = []
    total_f1 = 0.0
    total_em = 0.0

    # 5. Loop sugli esempi del test set
    for idx, (_, row) in enumerate(df.iterrows()):
        raw_messages = str(row["messages"])
        raw_answers = str(row["answers"])

        question = parse_question(raw_messages)
        gold = parse_answer(raw_answers)

        try:
            # Retrieval: cerco i contesti più rilevanti
            contexts = retriever.search(question, k=3)

            # Generation: chiedo a Gemini usando i contesti
            pred = generator.generate_answer(question, contexts)

        except Exception as e:
            # Se qualcosa va storto per questo sample, lo segniamo e andiamo avanti
            pred = f"ERRORE: {e}"
            contexts = []

        # Calcolo metriche
        f1 = f1_score(pred, gold)
        em = exact_match(pred, gold)

        total_f1 += f1
        total_em += em

        rows.append(
            {
                "question": question,
                "ground_truth_answer": gold,
                "generated_answer": pred,
                "f1": f1,
                "em": em,
                "contexts": " ||| ".join(contexts),
            }
        )

        # Piccolo debug sui primi esempi (utile per capire cosa sta succedendo)
        if idx < 3:
            print("\n--- ESEMPIO DEBUG ---")
            print("Q:", question)
            print("GOLD:", gold)
            print("PRED:", pred)
            print("F1, EM:", f1, em)

    # 6. Salvo il report completo
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORT_PATH, index=False)

    # 7. Stampo le metriche medie
    n = len(rows)
    avg_f1 = total_f1 / n if n else 0.0
    avg_em = total_em / n if n else 0.0

    print("\n✅ Valutazione conclusa.")
    print(f"📁 Report salvato in: {REPORT_PATH}")
    print(f"📊 F1 medio: {avg_f1:.4f}")
    print(f"📊 EM medio: {avg_em:.4f}")


if __name__ == "__main__":
    run_evaluation()
