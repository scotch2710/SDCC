import pandas as pd
import requests
import json
import time

# --- CONFIGURAZIONE ---
# 1. L'URL della tua API (Controlla il terminale quando premi F5)
API_URL = "http://localhost:7071/api/predict"

# 2. Il file di test che hai sul PC
TEST_FILE = "DIA_testset_RDKit_descriptors.csv"

# 3. Il nome della colonna con la risposta corretta (0 o 1)
# Ho controllato il tuo file: la colonna si chiama "Label"
TARGET_COLUMN = "Label" 

def test_pipeline():
    print(f"--- AVVIO TEST MODELLE ---")
    
    # A. Carichiamo il file
    try:
        df = pd.read_csv(TEST_FILE)
        print(f"File caricato: {len(df)} farmaci trovati.")
    except FileNotFoundError:
        print(f"ERRORE: Non trovo il file '{TEST_FILE}'.")
        return

    # Prendiamo un campione di 10 farmaci a caso per il test
    sample = df.sample(120)
    
    correct_predictions = 0
    total_tests = len(sample)

    print("\nInizio invio richieste all'API...\n")

    for index, row in sample.iterrows():
        # --- PREPARAZIONE DATI ---
        # 1. Salviamo il valore vero (la risposta corretta)
        real_label = row[TARGET_COLUMN]
        
        # 2. Prepariamo il pacchetto dati (Payload) da inviare
        # Dobbiamo togliere la risposta ('Label') perché l'API deve indovinarla
        # Dobbiamo togliere 'SMILES' perché è testo e rompe il modello
        input_data = row.drop([TARGET_COLUMN, 'SMILES'], errors='ignore')
        
        # Convertiamo in dizionario
        payload = input_data.to_dict()

        # --- CHIAMATA API ---
        try:
            # Inviamo la richiesta POST
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                # Leggiamo la risposta JSON dell'API
                api_response = response.json()
                
                # La predizione arriva come stringa ("0" o "1"), la convertiamo in intero
                prediction = int(float(api_response.get("prediction")))
                
                # --- VERIFICA ---
                if prediction == real_label:
                    esito = "✅ CORRETTO"
                    correct_predictions += 1
                else:
                    esito = "❌ ERRORE  "
                
                print(f"Farmaco #{index} -> Reale: {real_label} | Predetto: {prediction} | {esito}")
            
            else:
                print(f"Farmaco #{index} -> Errore API: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"Errore di connessione: {e}")

    # --- RISULTATO FINALE ---
    accuracy = (correct_predictions / total_tests) * 100
    print(f"\n--- TEST COMPLETATO ---")
    print(f"Accuratezza sul campione: {accuracy}% ({correct_predictions}/{total_tests})")

if __name__ == "__main__":
    test_pipeline()