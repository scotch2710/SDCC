import streamlit as st
import pandas as pd
import requests
import json

# --- CONFIGURAZIONE ---
# Se stai testando in locale usa localhost
# Se sei sul cloud usa il link https://...
API_URL = "https://sdcc-gallo-fabrizio-app-c9bjg6bbcsa4aph3.italynorth-01.azurewebsites.net/api/predict" 
TEST_FILE = "DIA_testset_RDKit_descriptors.csv"
TARGET_COLUMN = "Label"

# Impostiamo il titolo e la configurazione della pagina
st.set_page_config(page_title="SDCC Drug Toxicity AI", page_icon="💊", layout="wide")

st.title("💊 AI Drug Safety Prediction")
st.markdown("""
Questa dashboard permette di interrogare il modello di Machine Learning dispiegato su **Azure Functions**.
Il sistema analizza i descrittori chimici RDKit per predire la tossicità.
""")

# --- CARICAMENTO DATI ---
@st.cache_data # Mantiene i dati in memoria per non ricaricarli a ogni click
def load_data():
    try:
        df = pd.read_csv(TEST_FILE)
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error(f"Errore: Non trovo il file '{TEST_FILE}'. Assicurati che sia nella stessa cartella.")
    st.stop()

# --- SIDEBAR (Menu laterale) ---
st.sidebar.header("Pannello di Controllo")
mode = st.sidebar.radio("Modalità di Input", ["Seleziona da Test Set", "Inserimento Manuale (Simulato)"])

# --- LOGICA ---
input_data = None
real_label = None

if mode == "Seleziona da Test Set":
    st.subheader("📂 Seleziona un farmaco dal database")
    
    # Mostriamo un menu a tendina con gli indici dei farmaci
    drug_id = st.selectbox("Scegli l'ID del farmaco (indice riga):", df.index)
    
    # Estraiamo la riga selezionata
    row = df.loc[drug_id]
    
    # Salviamo il valore reale per il confronto (se esiste) e poi lo togliamo
    if TARGET_COLUMN in row:
        real_label = row[TARGET_COLUMN]
        # Togliamo Label e SMILES per l'invio all'API
        input_data = row.drop([TARGET_COLUMN, 'SMILES'], errors='ignore')
    else:
        input_data = row.drop(['SMILES'], errors='ignore')

    # Mostriamo i dati chimici a video
    st.write("### 🧪 Caratteristiche Chimiche (Preview)")
    # Mostriamo solo le prime 10 colonne per non intasare lo schermo
    st.dataframe(pd.DataFrame(input_data).T.head(10)) 
    st.caption(f"...e altre {len(input_data)-10} caratteristiche chimiche.")

elif mode == "Inserimento Manuale (Simulato)":
    st.subheader("✏️ Inserimento Manuale")
    st.info("Dato che ci sono >50 parametri chimici, partiamo da un farmaco base e puoi modificarne i valori principali.")
    
    # Prendiamo il primo farmaco come base
    base_id = st.number_input("ID Farmaco Base", min_value=0, max_value=len(df)-1, value=0)
    row = df.loc[base_id]
    
    # Creiamo dei controlli (slider) per modificare alcuni valori chiave
    # NOTA: Qui ho scelto 3 colonne a caso (MolWt, BalabanJ, BertzCT). 
    # Streamlit permette di modificarle al volo.
    col1, col2, col3 = st.columns(3)
    
    new_mol_wt = col1.number_input("Peso Molecolare (MolWt)", value=float(row.get("MolWt", 0.0)))
    new_balaban = col2.number_input("Indice BalabanJ", value=float(row.get("BalabanJ", 0.0)))
    new_bertz = col3.number_input("Indice BertzCT", value=float(row.get("BertzCT", 0.0)))
    
    # Prepariamo i dati aggiornando con i valori manuali
    input_data = row.drop([TARGET_COLUMN, 'SMILES'], errors='ignore')
    input_data["MolWt"] = new_mol_wt
    input_data["BalabanJ"] = new_balaban
    input_data["BertzCT"] = new_bertz
    
    st.write("Dati pronti per l'invio:")
    st.json({k: v for k, v in input_data.items() if k in ["MolWt", "BalabanJ", "BertzCT"]})

# --- BOTTONE DI PREDIZIONE ---
st.divider()
if st.button("🚀 Analizza Tossicità (Chiama API Azure)", type="primary"):
    
    with st.spinner("Invio dati al cloud in corso..."):
        try:
            # Convertiamo la Series pandas in dizionario
            payload = input_data.to_dict()
            
            # CHIAMATA HTTP
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = int(float(result.get("prediction")))
                
                # Visualizzazione Risultato
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if prediction == 1:
                        st.error("⚠️ RISULTATO: TOSSICO (Autoimmune Positive)")
                    else:
                        st.success("✅ RISULTATO: SICURO (Autoimmune Negative)")
                
                with col_res2:
                    if real_label is not None:
                        st.metric("Valore Reale nel Dataset", f"{real_label}", 
                                  delta="Corretto" if prediction == real_label else "Errore",
                                  delta_color="normal")
                
                st.success("Risposta ricevuta dall'API in Azure Functions.")
                st.json(result) # Mostra il JSON grezzo per fare scena tecnica
                
            else:
                st.error(f"Errore API: {response.status_code}")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Impossibile connettersi all'API. Assicurati che Azure Function sia attiva.\nErrore: {e}")