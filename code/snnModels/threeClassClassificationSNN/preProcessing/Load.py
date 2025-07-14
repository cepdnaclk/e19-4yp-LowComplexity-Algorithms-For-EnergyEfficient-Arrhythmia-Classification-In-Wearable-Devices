import wfdb
import os
from pathlib import Path

def load_ecg(record_id, data_dir):
    """
    Load ECG record and return signal, annotations, and sampling frequency.
    
    Args:
        record_id (str): MIT-BIH record number (e.g., '100')
        data_dir (str): Path to directory containing MIT-BIH data
    
    Returns:
        tuple: (signal, rpeaks, fs, ann) or None if loading fails
    """
    try:
        # Load record and annotations
        record = wfdb.rdrecord(f'{data_dir}/{record_id}')
        ann = wfdb.rdann(f'{data_dir}/{record_id}', 'atr')
        return record.p_signal[:, 0], ann.sample, record.fs, ann
    except FileNotFoundError:
        print(f"Error: Record {record_id} not found in {data_dir}")
        return None
    except Exception as e:
        print(f"Error loading record {record_id}: {str(e)}")
        return None

# Test code for load_ecg function with data got by https://physionet.org/lightwave/
# Explaination -> AS example the 101 th record is 30.06 mins long so it contain (30*60 + 6)*360 = 650160 samples (can be differ).
# The rpeaks array contains the sample number which is having a R peak. R peaks are identified bu the annotations

