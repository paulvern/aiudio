# Audio Master Pro - README

## Panoramica

Audio Master Pro è un'applicazione desktop avanzata per l'elaborazione audio professionale che combina tecnologie di intelligenza artificiale e DSP (Digital Signal Processing) per migliorare la qualità dell'audio. Con un'interfaccia moderna e intuitiva, offre strumenti potenti per il restauro, la pulizia e il mastering audio.

## Caratteristiche Principali

### Elaborazione AI
- **AI Quality Restorer**: Utilizza Demucs per separare e migliorare le componenti audio
- **Voice Cleaner**: Ottimizzato per la pulizia vocale con Facebook Denoiser
- **Fast Denoise**: Riduzione del rumore rapida basata su DSP

### Strumenti di Mastering
- **Equalizzatore a 10 bande** con preset professionali
- **Compressore dinamico** con controlli completi (threshold, ratio, attack, release)
- **Limiter** per massimizzare il volume senza distorsioni
- **Controllo stereo** per regolare l'ampiezza dell'immagine stereo
- **De-esser** per ridurre le sibilanti nelle registrazioni vocali
- **Effetti creativi**: Riverbero e saturazione armonica

### Visualizzazione Avanzata
- **Waveform display** interattivo per input e output
- **Analizzatore di spettro** in tempo reale
- **Indicatori di livello** per monitorare l'audio

### Player Professionale
- **Controlli A-B** per confrontare sezioni specifiche
- **Regolazione della velocità** di riproduzione
- **Loop** per ripetere sezioni

### Altre Funzionalità
- **Esportazione** in formati WAV, MP3 e FLAC
- **Interfaccia moderna** con tema scuro professionale
- **Supporto per diversi formati** audio in ingresso

## Requisiti di Sistema

- **Sistema Operativo**: Windows 10/11
- **RAM**: 8GB minimo (16GB raccomandati)
- **Processore**: Intel i5/AMD Ryzen 5 o superiore
- **GPU**: Supporto CUDA (opzionale, per accelerazione AI)
- **Spazio su disco**: 2GB per l'installazione

## Installazione

1. Assicurati di avere Python 3.10 o 3.11 installato
2. Crea un ambiente virtuale:
   ```
   python -m venv venv
   ```
3. Attiva l'ambiente virtuale:
   ```
   venv\Scripts\activate
   ```
4. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```
5. Avvia l'applicazione:
   ```
   python aiudio.py
   ```

## Guida Rapida

### Elaborazione Base
1. Carica un file audio con il pulsante "Load Audio"
2. Seleziona la modalità di elaborazione AI desiderata
3. Clicca "PROCESS" per avviare l'elaborazione
4. Usa i controlli DSP per perfezionare il risultato
5. Esporta il risultato con "EXPORT RESULT"

### Utilizzo dell'Equalizzatore
- Seleziona un preset dal menu a tendina o regola manualmente le bande
- Le frequenze vanno da 32Hz a 16kHz per coprire l'intero spettro udibile
- Usa valori positivi per enfatizzare, negativi per attenuare

### Compressione
- Imposta la soglia (Threshold) per determinare quando inizia la compressione
- Regola il rapporto (Ratio) per controllare l'intensità della compressione
- Usa Attack e Release per definire la risposta temporale

### Effetti
- Abilita il riverbero per aggiungere spazialità
- Usa la saturazione per aggiungere calore e carattere
- Regola la larghezza stereo per espandere o restringere l'immagine sonora

### Player
- Usa i controlli di riproduzione per ascoltare l'audio originale o elaborato
- Imposta punti A-B per riprodurre solo sezioni specifiche
- Regola volume e velocità secondo necessità

## Risoluzione Problemi

### L'applicazione non si avvia
- Verifica che tutte le dipendenze siano installate correttamente
- Assicurati di usare Python 3.10 o 3.11
- Controlla che l'ambiente virtuale sia attivato

### Errori durante l'elaborazione AI
- Verifica la connessione internet (necessaria per il download iniziale dei modelli)
- Assicurati di avere spazio su disco sufficiente
- Se hai una GPU, verifica che i driver CUDA siano aggiornati

### Audio distorto dopo l'elaborazione
- Riduci il guadagno di output
- Disattiva il limiter o aumenta il valore di ceiling
- Riduci i valori dell'equalizzatore, specialmente nelle basse frequenze

## Crediti e Tecnologie

Audio Master Pro utilizza diverse librerie open source:
- **Demucs** per la separazione delle sorgenti audio
- **PyTorch** per l'elaborazione AI
- **Librosa** per l'analisi audio
- **PyQt5** per l'interfaccia grafica
- **Matplotlib** per la visualizzazione
- **SoundDevice** per la riproduzione audio
- **SoundFile** per l'I/O audio
- **NoiseReduce** per la riduzione del rumore basata su DSP

## Licenza

Audio Master Pro è distribuito sotto licenza MIT. Vedi il file LICENSE per i dettagli.

---

© 2024 Audio Master Pro. Tutti i diritti riservati.
