# src/main.py

import os

from src.dataprocessing import DocumentProcessor
from src.retrieval import Retriever
from src.generation import Generator


def ensure_vector_store(retriever: Retriever):
    """
    Si assicura che l'indice vettoriale (FAISS + chunks) esista.
    - Se esiste: lo carica.
    - Se non esiste: lo costruisce dai documenti CSV.
    """
    index_exists = hasattr(retriever, "index_path") and os.path.exists(retriever.index_path)
    chunks_exists = hasattr(retriever, "chunks_path") and os.path.exists(retriever.chunks_path)

    if index_exists and chunks_exists:
        print("📁 Vector store trovato. Carico indice e chunks da disco...")
        retriever.load_index()
    else:
        print("⚠️ Nessun vector store trovato. Lo costruisco dai documenti CSV...")
        processor = DocumentProcessor()
        docs = processor.load_documents_from_csv()
        print(f"📄 Documenti caricati dal CSV: {len(docs)}")

        chunks = processor.split_documents(docs)
        print(f"🔹 Chunk generati: {len(chunks)}")

        retriever.build_index(chunks)
        print("✅ Vector store creato.")


def interactive_rag():
    """
    Modalità interattiva:
    1. Inizializza Retriever + Generator.
    2. Garantisce che l'indice esista.
    3. Permette all'utente di fare domande e vedere:
       - risposta generata (Gemini)
       - contesti usati (per debug/trasparenza).
    """
    print("🚀 Avvio RAG DMV interattivo...")

    # Inizializza retriever e vector store
    retriever = Retriever()
    ensure_vector_store(retriever)

    # Inizializza generatore (Gemini 2.0 Flash via API)
    generator = Generator()

    print("\n💬 Sistema pronto.")
    print("Scrivi una domanda sul DMV (o 'exit' per uscire).")

    while True:
        query = input("\n👉 Domanda: ").strip()

        if query.lower() in ("exit", "quit"):
            print("👋 Uscita dal RAG.")
            break

        if not query:
            print("⚠️ Inserisci una domanda non vuota.")
            continue

        # 1️⃣ Retrieval: cerca i chunk più rilevanti
        try:
            contexts = retriever.search(query, k=3)
        except Exception as e:
            print(f"❌ Errore nel retrieval: {e}")
            continue

        if not contexts:
            print("⚠️ Nessun contesto trovato per questa domanda.")
            continue

        # 2️⃣ Generation: genera risposta usando i contesti
        try:
            answer = generator.generate_answer(query, contexts)
        except Exception as e:
            print(f"❌ Errore nella generazione con Gemini: {e}")
            continue

        # Output
        print("\n🤖 Risposta:")
        print(answer)

        # (Opzionale ma didattico) Mostra i contesti usati
        print("\n📚 Contesti utilizzati (per capire cosa ha letto il modello):")
        for i, ctx in enumerate(contexts, start=1):
            anteprima = ctx[:400].replace("\n", " ")
            print(f"\n--- Context {i} ---")
            print(anteprima + ("..." if len(ctx) > 400 else ""))


if __name__ == "__main__":
    interactive_rag()
