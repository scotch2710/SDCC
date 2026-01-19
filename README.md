Realizzare una pipeline automatizzata per il Machine Learning, basata su architettura serverless e tecnologie FaaS. Il flusso deve iniziare con il caricamento di un file contenente dati da analizzare dal seguente dataset pubblico su un servizio di storage sul cloud: Drug induced Autoimmunity Dataset

STEP 1: Il caricamento del dataset sullo storage deve fungere da evento trigger e attivare automaticamente una funzione FaaS responsabile della fase successiva. 

STEP 2: In questa fase, i dati vengono letti, analizzati, trasformati per renderli idonei per l’analisi.  Si effettuano, ad esempio, operazioni come la rimozione di valori nulli, la normalizzazione delle feature, la codifica di variabili categoriche, e altre trasformazioni necessarie per la qualità del dataset. Al termine di questa fase, i dati puliti e trasformati vengono salvati in una nuova posizione all'interno dello storage

STEP 3: a questo punto, una seconda funzione serverless viene attivata (automaticamente o tramite schedulazione). Questa funzione ha il compito di addestrare un modello di Machine Learning utilizzando i dati processati. Il tipo di modello può essere scelto tra algoritmi comunemente usati per quel dataset (es. regressione logistica, decision tree, random forest, support vector machine, ecc.) e viene addestrato utilizzando una libreria standard per il machine learning (es. scikit-learn). Una volta completato l’addestramento, il modello viene serializzato e salvato in una posizione dedicata all’interno dello storage.

STEP 4: questa fase implementa il servizio di inferenza e può essere implementata tramite un’ulteriore funzione FaaS o VM che espone un’interfaccia HTTP. Tale endpoint consente di inviare nuove istanze di dati per ottenere predizioni in tempo reale. 
