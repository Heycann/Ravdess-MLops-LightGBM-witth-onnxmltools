import os
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"  # ⚠️ Wajib sebelum import torchaudio
import warnings
warnings.filterwarnings("ignore")

import torchaudio
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping RAVDESS ke Stress Level (Arousal-based)
EMOTION_TO_STRESS = {
    '01': 0,  # Neutral -> LOW
    '02': 0,  # Calm -> LOW
    '03': 1,  # Happy -> MEDIUM
    '04': 1,  # Sad -> MEDIUMs
    '07': 1,  # Disgust -> MEDIUM
    '05': 2,  # Angry -> HIGH
    '06': 2,  # Fearful -> HIGH
    '08': 2   # Surprised -> HIGH
}

def parse_ravdess_label(filename: str) -> str:
    """Ekstrak kode emosi dari format RAVDESS: 03-01-[05]-01-01-01-01.wav"""
    parts = Path(filename).stem.split('-')
    if len(parts) == 7:
        return parts[2] # Mengambil bagian ke-3 (kode emosi)
    return '01' # Default netral jika format salah

def extract_chunk_features(y_chunk: np.ndarray, sr: int) -> np.ndarray:
    """Ekstrak fitur statistik per chunk 0.5s"""
    mfcc = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    rms = librosa.feature.rms(y=y_chunk)[0]
    zcr = librosa.feature.zero_crossing_rate(y_chunk)[0]
    spec_cent = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr)[0]
    
    f0, _, _ = librosa.pyin(y_chunk, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    f0 = np.nan_to_num(f0, nan=0.0)
    
    def agg(x): 
        if len(x) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [np.mean(x), np.std(x), np.min(x), np.max(x)]
    
    feats = []
    for arr in [mfcc, delta, delta2]:
        for i in range(arr.shape[0]):
            feats.extend(agg(arr[i]))
    for arr in [rms, zcr, spec_cent, spec_rolloff, f0]:
        feats.extend(agg(arr))
        
    return np.array(feats)

def process_dataset(raw_dir: str, out_dir: str, chunk_sec: float = 0.5, hop_sec: float = 0.25, sr: int = 16000):
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"❌ Folder {raw_dir} tidak ditemukan!")

    chunk_len = int(chunk_sec * sr) 
    hop_len = int(hop_sec * sr)
    
    # Cari rekursif (.wav huruf kecil dan besar)
    wav_files = list(raw_path.rglob('*.wav')) + list(raw_path.rglob('*.WAV'))
    
    if not wav_files:
        logging.warning(f"⚠️ Tidak ada .wav di {raw_dir}. Isi folder saat ini:")
        for p in list(raw_path.iterdir())[:10]:
            logging.warning(f"  - {p.name}")
        raise FileNotFoundError(f"Tidak ada file .wav di {raw_dir}.")
        
    logging.info(f"✅ Ditemukan {len(wav_files)} file .wav RAVDESS")
    
    records = []
    for wav in tqdm(wav_files):
        emotion_code = parse_ravdess_label(wav.name)
        stress_label = EMOTION_TO_STRESS.get(emotion_code, 0)
        
        try:
            wav_str = str(wav) 
            y, orig_sr = sf.read(wav_str, dtype='float32')
            
            if y.ndim > 1:
                y = y.mean(axis=1)

            if orig_sr != sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
                
        except Exception as e:
            logging.warning(f"Gagal load {wav.name}: {repr(e)}")
            continue
            
        # Chunking untuk real-time compatibility
        for start in range(0, len(y) - chunk_len + 1, hop_len):
            chunk = y[start:start+chunk_len]

            if np.max(np.abs(chunk)) < 1e-5:  
                continue
                
            feats = extract_chunk_features(chunk, sr)
            records.append({
                'filename': wav.name,
                'chunk_start_sec': start / sr,
                'stress_label': stress_label,
                'features': feats
            })

    if not records:
        raise RuntimeError(
            "❌ Tidak ada chunk audio yang berhasil diproses.\n"
            "Penyebab: semua file gagal load, threshold silence terlalu tinggi, atau path salah."
        )

    logging.info("Membuat DataFrame dan menyusun fitur...")
    df = pd.DataFrame(records)
    
    feat_cols = [f'feat_{i}' for i in range(len(df['features'].iloc[0]))]
    features_df = pd.DataFrame(df['features'].tolist(), index=df.index, columns=feat_cols)
    df = pd.concat([df.drop(columns=['features']), features_df], axis=1)
    
    logging.info("Membagi dataset (Train/Val/Test)...")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['stress_label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['stress_label'], random_state=42)
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Menyimpan ke file Parquet...")
    train_df.to_parquet(out_path / 'train.parquet', index=False)
    val_df.to_parquet(out_path / 'val.parquet', index=False)
    test_df.to_parquet(out_path / 'test.parquet', index=False)
    
    logging.info(f"✅ Selesai. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RAVDESS Preprocessing Pipeline for MLOps")
    parser.add_argument('--raw', default='data/raw/RAVDESS', help="Path ke folder dataset mentah")
    parser.add_argument('--out', default='data/processed', help="Path output untuk file parquet")
    args = parser.parse_args()
    
    process_dataset(args.raw, args.out)
    
    