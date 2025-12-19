import logging
import azure.functions as func
import pandas as pd
import io
import os
from azure.storage.blob import BlobServiceClient

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
        
        # 2. Codifica variabili categoriche (One-Hot Encoding)
        # Trasforma colonne di testo in numeri (es. "Maschio/Femmina" -> 0/1)
        # drop_first=True evita la collinearità
        df_clean = pd.get_dummies(df_clean, drop_first=True)
        
        logging.info(f"Dataset pulito. Dimensioni finali: {df_clean.shape}")

        # --- SALVATAGGIO SU STORAGE (processed-data) --- [cite: 10]
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