import wfdb

def load_ecg(record_id, data_dir):
    """
    Load ECG record and return signal, annotations, and sampling frequency
    
    Args:
        record_id: MIT-BIH record number (e.g., '100')
        data_dir: Path to directory containing the MIT-BIH data
    
    Returns:
        signal: ECG signal (Lead II)
        rpeaks: Sample indices of R-peaks -> Total beats in  101  record :  [     7     83    396 ... 649004 649372 649751]
        fs: Sampling frequency (Hz)
    """
    # Load record and annotations 
    record = wfdb.rdrecord(f'{data_dir}/{record_id}')
    ann = wfdb.rdann(f'{data_dir}/{record_id}', 'atr')
    return record.p_signal[:, 0], ann.sample, record.fs, ann

# Test code for load_ecg function with data got by https://physionet.org/lightwave/
# if __name__ == "__main__":
#     signal, events, fs, ann = load_ecg('108', 'data/mitdb')
#     print("No of events : ", len(events))
#     print("No of annotations : ", len(ann.sample))
#     print("First 10 samples related to events : ", ann.sample[:10])
#     print("First 10 annotations/events : ", ann.symbol[:10])

# Explaination -> AS example the 101 th record is 30.06 mins long so it contain (30*60 + 6)*360 = 650160 samples (can be differ).
# The rpeaks array contains the sample number which is having a R peak. R peaks are identified bu the annotations

