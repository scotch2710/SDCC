import logging
import azure.functions as func
import pandas as pd
import io
import os
from azure.storage.blob import BlobServiceClient
import joblib
import json
from sklearn.ensemble import RandomForestClassifier

app = func.FunctionApp()

# CONFIGURAZIONE: Questa funzione scatta quando un file entra in "input-data"
@app.blob_trigger(arg_name="myblob", path="input-data/{name}", connection="AzureWebJobsStorage")
def data_preprocessing(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name} \n"
                 f"Blob Size: {myblob.length} bytes")

    try:
        # --- STEP 1: LETTURA DATI --- 
        # Leggiamo il contenuto del file caricato (CSV) in memoria
        file_content = myblob.read()
        
        # Trasformiamo i byte in un DataFrame Pandas
        # Usiamo io.BytesIO perché pandas si aspetta un file-like object
        df = pd.read_csv(io.BytesIO(file_content))
        logging.info(f"Dataset caricato. Dimensioni originali: {df.shape}")

        # --- STEP 2: PREPROCESSING --- 
        # Eseguiamo le operazioni richieste dal progetto:
        
        # 1. Rimozione valori nulli
        df_clean = df.dropna()

        # 2. Rimozione colonna SMILES
        # La formula chimica testuale non serve al modello matematico, 
        # usiamo solo i descrittori numerici già calcolati.
        if "SMILES" in df_clean.columns:
            df_clean = df_clean.drop(columns=["SMILES"])
        
        # 2. Codifica variabili categoriche 
        # Trasforma colonne di testo in numeri (es. "Maschio/Femmina" -> 0/1)
        # drop_first=True evita la collinearità
        df_clean = pd.get_dummies(df_clean, drop_first=True)
        
        logging.info(f"Dataset pulito. Dimensioni finali: {df_clean.shape}")

        # --- SALVATAGGIO SU STORAGE (processed-data) 
        # Prepariamo il buffer per salvare il CSV
        output_buffer = io.StringIO()
        df_clean.to_csv(output_buffer, index=False)
        
        # Recuperiamo la stringa di connessione dalle variabili d'ambiente
        connect_str = os.getenv('AzureWebJobsStorage')
        
        # Creiamo il client per connetterci allo storage
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Definiamo il nome del file di output (stesso nome dell'input)
        input_filename = os.path.basename(myblob.name)
        container_client = blob_service_client.get_container_client("processed-data")
        
        # Carichiamo il file processato nel container "processed-data"
        container_client.upload_blob(name=input_filename, data=output_buffer.getvalue(), overwrite=True)
        
        logging.info(f"File {input_filename} salvato correttamente in 'processed-data'")

    except Exception as e:
        logging.error(f"Errore durante l'elaborazione: {e}")
        raise e
    

# --- CONFIGURAZIONE GLOBALE ---

TARGET_COLUMN = "Label"   #colonna da predire

# --- STEP 3: TRAINING DEL MODELLO ---
# Questa funzione parte automaticamente quando un file pulito arriva in 'processed-data'
@app.blob_trigger(arg_name="myblob", path="processed-data/{name}", connection="AzureWebJobsStorage")
def train_model(myblob: func.InputStream):
    logging.info(f"--- INIZIO TRAINING ---")
    logging.info(f"File rilevato: {myblob.name}")

    try:
        # 1. Lettura del Dataset Processato
        # Leggiamo i dati puliti dallo storage
        file_content = myblob.read()
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Verifica di sicurezza: controlliamo se la colonna target esiste
        if TARGET_COLUMN not in df.columns:
            logging.error(f"ERRORE CRITICO: La colonna '{TARGET_COLUMN}' non esiste nel file! Controlla il CSV.")
            return

        # 2. Preparazione Dati (X e y)
        # X: Tutte le colonne tranne il target (sono le caratteristiche chimiche)
        X = df.drop(columns=[TARGET_COLUMN])
        # y: Solo la colonna target (0 o 1)
        y = df[TARGET_COLUMN]
        
        logging.info(f"Training su {len(df)} righe e {len(X.columns)} features.")

        # 3. Addestramento (Random Forest)
        # Usiamo 100 alberi decisionali per una buona accuratezza
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        logging.info("Modello addestrato con successo.")

        # 4. Serializzazione e Salvataggio
        # Salviamo il modello in un buffer di memoria come file .pkl
        model_filename = "model.pkl"
        model_buffer = io.BytesIO()
        joblib.dump(clf, model_buffer)
        
        # Connessione allo storage per salvare il file finale
        connect_str = os.getenv('AzureWebJobsStorage')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Carichiamo il modello nel container 'models'
        container_client = blob_service_client.get_container_client("models")
        container_client.upload_blob(name=model_filename, data=model_buffer.getvalue(), overwrite=True)
        
        logging.info(f"SUCCESS: Modello salvato come '{model_filename}' nel container 'models'.")

    except Exception as e:
        logging.error(f"ERRORE TRAINING: {e}")
        raise e


# --- STEP 4: API (PREDIZIONE) ---
# Questa funzione espone un indirizzo HTTP per ricevere dati e dare risposte
@app.route(route="predict", auth_level=func.AuthLevel.ANONYMOUS)
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Richiesta di predizione ricevuta.')

    try:
        # 1. Parsing della richiesta (Input JSON)
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse("Errore: Il corpo della richiesta deve essere un JSON valido.", status_code=400)

        # Trasformiamo il JSON ricevuto in un DataFrame (come se fosse una riga di CSV)
        # Assumiamo che il JSON contenga già i descrittori RDKit corretti
        input_data = pd.DataFrame([req_body])

        # 2. Recupero del Modello dallo Storage
        logging.info("Scaricamento del modello da Azure Storage...")
        connect_str = os.getenv('AzureWebJobsStorage')
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        
        # Puntiamo al file 'model.pkl' nel container 'models'
        blob_client = blob_service_client.get_blob_client(container="models", blob="model.pkl")
        
        if not blob_client.exists():
             return func.HttpResponse("Errore: Modello non ancora addestrato (file model.pkl mancante).", status_code=500)

        # Scarichiamo il modello in memoria
        downloader = blob_client.download_blob()
        model_buffer = io.BytesIO(downloader.readall())
        
        # 3. Caricamento e Predizione
        loaded_model = joblib.load(model_buffer)
        
        # Il modello potrebbe aver bisogno che le colonne siano nello stesso ordine del training.
        
        
        prediction = loaded_model.predict(input_data)
        
        # Risultato: 0 o 1 (o la classe originale)
        result_value = prediction[0]
        
        # 4. Risposta al Client
        response_payload = {
            "prediction": str(result_value),
            "status": "success",
            "message": "Analisi completata."
        }
        
        return func.HttpResponse(
            json.dumps(response_payload),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"ERRORE INFERENZA: {str(e)}")
        return func.HttpResponse(
             f"Errore interno del server: {str(e)}",
             status_code=500
        )