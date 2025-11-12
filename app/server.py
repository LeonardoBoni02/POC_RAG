import os
import sys
from flask import Flask, render_template, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

# Aggiungi la cartella 'src' al path di Python per permettere le importazioni
# Questo è necessario perché stiamo eseguendo da 'app/server.py' ma i moduli sono in 'src/'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.dataprocessing import DocumentProcessor
    from src.retrieval import Retriever
    from src.generation import Generator
except ImportError:
    print("Errore: Impossibile importare i moduli da /src.")
    print("Assicurati che 'src' sia nella root del progetto e contenga __init__.py (anche se vuoto)")
    sys.exit(1)


# --- Funzione di Setup (presa da main.py) ---

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
        
        if not docs:
            print("❌ Dati CSV non trovati o vuoti. L'indicizzazione non può continuare.")
            print("Assicurati che 'src/filtered_data_code.py' sia stato eseguito correttamente.")
            return False # Segnala fallimento

        print(f"📄 Documenti caricati dal CSV: {len(docs)}")

        chunks = processor.split_documents(docs)
        print(f"🔹 Chunk generati: {len(chunks)}")

        retriever.build_index(chunks)
        print("✅ Vector store creato.")
    return True # Segnala successo

# --- Inizializzazione App ---

print("🚀 Avvio RAG DMV Web App...")
app = Flask(__name__, template_folder='templates')
metrics = PrometheusMetrics(app) # Attiva il monitoring /metrics

# Oggetti globali per RAG
retriever = None
generator = None
app_ready = False

try:
    print("Inizializzazione Retriever (FAISS)...")
    retriever = Retriever()
    
    print("Inizializzazione Generator (Gemini)...")
    generator = Generator() # Questo controllerà la GOOGLE_API_KEY
    
    print("Verifica Vector Store...")
    app_ready = ensure_vector_store(retriever)
    
    if app_ready:
        print("✅ Applicazione pronta a ricevere richieste.")
    else:
        print("❌ Errore critico durante l'avvio. Controllare i log.")

except Exception as e:
    print(f"❌ Errore fatale durante l'inizializzazione: {e}")
    print("L'applicazione potrebbe non funzionare. Controlla la GOOGLE_API_KEY e i percorsi.")

# --- Endpoints ---

@app.route('/', methods=['GET'])
def index():
    """Serve la pagina HTML principale."""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """Esegue la pipeline RAG (Retrieve + Generate)."""
    
    if not app_ready or retriever is None or generator is None:
        return jsonify({"error": "Applicazione non ancora pronta o in stato di errore."}), 503

    query = request.form.get('text', '')
    if not query:
        return jsonify({"error": "Nessuna domanda fornita."}), 400

    try:
        # 1. Retrieval
        contexts = retriever.search(query, k=3)
        
        if not contexts:
            print(f"⚠️ Nessun contesto trovato per: '{query}'")
            # Possiamo decidere di rispondere comunque o solo con il LLM
            # Per ora, seguiamo il prompt originale
        
        # 2. Generation
        answer = generator.generate_answer(query, contexts)

        return jsonify({
            "query": query,
            "answer": answer,
            "contexts": contexts
        })

    except Exception as e:
        print(f"Errore durante l'elaborazione della richiesta: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Usato solo per test locale, in produzione useremo Gunicorn
    app.run(host='0.0.0.0', port=8000, debug=False)