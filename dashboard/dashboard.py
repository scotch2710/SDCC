import streamlit as st
import pandas as pd
import requests
import json
import os
import time
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()           # per leggere file nascosto che contiene chiave di connessione

# --- CONFIGURAZIONE ---

API_URL = "https://sdcc-gallo-fabrizio-app-c9bjg6bbcsa4aph3.italynorth-01.azurewebsites.net/api/predict" 
CONTAINER_INPUT = "input-data"


CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

if not CONNECTION_STRING:
    st.error("⚠️ ERRORE CRITICO: Connection String non trovata! Configura le variabili d'ambiente.")
    st.stop() # Ferma l'app se manca la chiave
# --- UI SETUP ---
st.set_page_config(page_title="SDCC Drug AI", page_icon="💊", layout="centered")

st.title("AI Drug Safety Platform")
st.markdown("### Dashboard di gestione per la predizione della tossicità")

# =========================================================
# SEZIONE 1: TEST & PREDIZIONE
# =========================================================
st.header("1. Area di Test (Predizione Singola)")
st.markdown("Carica un file CSV di test per analizzare i farmaci, oppure usa il dataset di default.")

# WIDGET: Caricamento file di test
test_file_buffer = st.file_uploader("📂 Carica il tuo file di Test (CSV)", type=["csv"], key="test_uploader")

@st.cache_data
def load_default_data():
    # Prova a caricare il file locale se esiste
    default_file = "DIA_testset_RDKit_descriptors.csv"
    try:
        return pd.read_csv(default_file)
    except FileNotFoundError:
        return None

# LOGICA DI CARICAMENTO
df = None
if test_file_buffer is not None:
    try:
        df = pd.read_csv(test_file_buffer)
        st.success(f"✅ Usando il file caricato: {test_file_buffer.name}")
    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
else:
    df = load_default_data()
    if df is not None:
        st.info("Usando il dataset di default (nessun file caricato).")

# SE ABBIAMO DEI DATI
if df is not None:
    # Menu a tendina per scegliere il farmaco
    drug_id = st.selectbox("Seleziona il farmaco da analizzare (riga):", df.index)
    
    # Estrazione dati della riga selezionata
    row = df.loc[drug_id]
    
    # Pulizia
    cols_to_drop = ['Label', 'SMILES', 'Unnamed: 0']
    input_data = row.drop([c for c in cols_to_drop if c in row.index], errors='ignore')
    
    with st.expander("🔍 Vedi parametri chimici (Input API)"):
        st.json(input_data.to_dict())

    if st.button("Analizza Tossicità", type="primary"):
        with st.spinner("Chiamata al Cloud Azure in corso..."):
            try:
                payload = input_data.to_dict()
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()

                    #with st.expander("🔍 DEBUG - Risposta Grezza API"):
                        #st.write(result)
                    
                    pred = result.get('prediction', -1)
                    
                    
                    #st.divider()
                    col1, col2 = st.columns(2)
                    
                    if str(pred) == "1":
                        col1.error("⚠️ RISULTATO: PERICOLOSO")
                    else:
                        col1.success("✅ RISULTATO: SICURO")
                        
                    # col2.metric("Confidenza Modello", f"{prob:.2f}%")
                else:
                    st.error(f"Errore API: {response.status_code}")
                    st.code(response.text)
                    
            except Exception as e:
                st.error(f"Errore di connessione: {e}")
else:
    st.warning("⚠️ Nessun dato disponibile. Carica un file CSV per iniziare.")

st.markdown("---")

# =========================================================
# SEZIONE 2: ANALISI MASSIVA (MODIFICATA)
# =========================================================
st.header("2. Analisi Massiva (Batch Testing)")
st.markdown("Esegui il test su un intervallo o su **tutto** il dataset per calcolare l'accuratezza.")

if df is not None:
    # Dividiamo in due colonne per le due opzioni (Intervallo vs Tutto)
    col_opt1, col_opt2 = st.columns(2)
    
    subset_to_analyze = None
    
    # OPZIONE A: INTERVALLO
    with col_opt1:
        st.subheader("Opzione A: Intervallo")
        max_idx = len(df) - 1
        start_idx = st.number_input("Indice Inizio", 0, max_idx, 0)
        end_idx = st.number_input("Indice Fine", start_idx, max_idx, min(start_idx + 5, max_idx))
        
        btn_range = st.button("▶️ Analizza Intervallo")

    # OPZIONE B: TUTTO IL DATASET
    with col_opt2:
        st.subheader("Opzione B: Completo")
        st.write(f"Totale farmaci nel file: **{len(df)}**")
        #st.warning("⚠️ L'analisi completa può richiedere tempo.")
        
        btn_all = st.button("Analizza TUTTO il Dataset")

    # LOGICA DI SELEZIONE
    if btn_range:
        subset_to_analyze = df.iloc[start_idx : end_idx + 1]
    elif btn_all:
        subset_to_analyze = df
    
    # ESECUZIONE ANALISI (Comune a entrambi i bottoni)
    if subset_to_analyze is not None:
        
        results_list = []
        correct_count = 0
        total = len(subset_to_analyze)
        
        st.write(f"Inizio analisi su **{total}** elementi...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (idx, batch_row) in enumerate(subset_to_analyze.iterrows()):
            # Aggiorna progress bar
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Analisi riga {idx} ({i+1}/{total})...")
            
            # Prepara dati
            b_input = batch_row.drop(['Label', 'SMILES', 'Unnamed: 0'], errors='ignore').to_dict()
            real_label = batch_row.get('Label', None)
            
            try:
                # Chiamata API
                resp = requests.post(API_URL, json=b_input)
                outcome = "Errore"
                pred_val = -1
                
                if resp.status_code == 200:
                    r_json = resp.json()
                    pred_val = r_json.get('prediction', -1)
                    
                    # Verifica correttezza
                    match = False
                    if real_label is not None:
                        if str(pred_val) == str(int(real_label)):
                            match = True
                            correct_count += 1
                        outcome = "✅ Corretto" if match else "❌ Errato"
                    else:
                        outcome = "N/A (No Label)"
                
                results_list.append({
                    "ID Riga": idx,
                    "Label Reale": int(real_label) if real_label is not None else "N/A",
                    "Predizione AI": pred_val,
                    "Esito": outcome
                })
                
            except Exception as e:
                results_list.append({"ID Riga": idx, "Esito": "Errore Conn"})
            
            # Pausa minima per stabilità
            time.sleep(0.02) 
            
        status_text.text("Analisi Completata!")
        
        # Mostra Metriche Finali
        if total > 0:
            acc = (correct_count / total) * 100
            st.success(f"🎯 Accuratezza Finale: **{acc:.2f}%** ({correct_count}/{total} corretti)")
        
        # Tabella Risultati
        with st.expander("Vedi Tabella Risultati Completa", expanded=True):
            res_df = pd.DataFrame(results_list)
            st.dataframe(res_df, use_container_width=True)

else:
    st.warning("Carica un dataset nel punto 1 per abilitare il Batch Testing.")

st.markdown("---")

# =========================================================
# SEZIONE 3: UPLOAD PER TRAINING
# =========================================================
st.header("3. Aggiornamento Modello (Training)")
st.markdown("""
Questa sezione permette di caricare nuovi dati grezzi nel Cloud. 
**Attenzione:** Il caricamento avvierà automaticamente la pipeline di ri-addestramento su Azure.
""")

# WIDGET: Caricamento file di training
train_file_buffer = st.file_uploader("Carica nuovi dati di Training (CSV)", type=["csv"], key="train_uploader")

if train_file_buffer is not None:
    # Pulsante di conferma per evitare upload accidentali
    if st.button("Conferma Upload su Azure Storage", type="secondary"):
        
        if "AccountKey" not in CONNECTION_STRING and "UseDevelopmentStorage=true" not in CONNECTION_STRING:
             st.error("⚠️ Connection String mancante o non valida nel codice.")
        else:
            with st.spinner(f"Sto caricando '{train_file_buffer.name}' nel container '{CONTAINER_INPUT}'..."):
                try:
                    # Connessione al Blob Storage
                    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
                    blob_client = blob_service_client.get_blob_client(container=CONTAINER_INPUT, blob=train_file_buffer.name)
                    
                    # Upload (overwrite=True sovrascrive se esiste già un file con lo stesso nome)
                    blob_client.upload_blob(train_file_buffer, overwrite=True)
                    
                    st.success("✅ Upload completato con successo!")
                    st.balloons()
                    st.markdown("Wait for the Azure Function trigger execution logs in the Azure Portal.")
                    
                except Exception as e:
                    st.error(f"Errore Upload: {e}")