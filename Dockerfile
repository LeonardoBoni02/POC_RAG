# Usa un'immagine Python 3.10 slim come base
FROM python:3.10-slim

# Imposta la directory di lavoro dentro il container
WORKDIR /app

# Installa dipendenze di sistema (necessarie per FAISS/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia prima il file dei requisiti
COPY requirements.txt .

# Installa le dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice dell'applicazione
COPY . .

# Rendi eseguibile lo script di avvio
RUN chmod +x /app/startup.sh

# Crea la directory 'data' dove verranno salvati CSV e indice FAISS
# Questo sarà un "mount point" per il volume Docker
RUN mkdir -p /app/data

# Esponi la porta 8000 (dove Gunicorn sarà in ascolto)
EXPOSE 8000

# Script da eseguire all'avvio del container
ENTRYPOINT ["/app/startup.sh"]