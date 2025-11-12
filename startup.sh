#!/bin/bash

# Naviga nella directory dell'app
cd /app

echo "--- 1. Esecuzione script di setup dati (filtered_data_code.py) ---"
# Esegue lo script che scarica/filtra il CSV (se non esiste già)
python3 src/filtered_data_code.py

echo "--- 2. Avvio del server Gunicorn sulla porta 8000 ---"
# Avvia il server web Gunicorn
# 'app.server:app' significa: "nel file 'app/server.py', trova l'oggetto 'app' (Flask)"
exec gunicorn --bind 0.0.0.0:8000 --workers 2 app.server:app