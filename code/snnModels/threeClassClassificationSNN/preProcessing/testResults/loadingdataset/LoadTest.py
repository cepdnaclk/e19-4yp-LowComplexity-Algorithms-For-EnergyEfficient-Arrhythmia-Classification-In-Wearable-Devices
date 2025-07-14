import wfdb
import pandas as pd
import os
from pathlib import Path
from Load import load_ecg

def get_dataset_summary(data_dir):
    """
    Summarize details of all records in the MIT-BIH dataset.
    
    Args:
        data_dir (str): Path to directory containing MIT-BIH data
    
    Returns:
        pd.DataFrame: Summary table with record details
    """
    # List of MIT-BIH record IDs (48 records)
    record_ids = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    # Initialize lists to store summary data
    summaries = []
    
    for record_id in record_ids:
        result = load_ecg(record_id, data_dir)
        if result is None:
            continue
        
        signal, rpeaks, fs, ann = result
        
        # Calculate record details
        num_samples = len(signal)
        duration_sec = num_samples / fs
        duration_min = duration_sec / 60
        num_annotations = len(ann.sample)
        
        beat_types = ann.symbol
        class_counts = {
            'Normal': sum(1 for s in beat_types if s in ['N']),
            'SupraventricularAndVentricular': sum(1 for s in beat_types if s in ['A', 'a', 'J', 'S', 'e', 'j', 'V', 'E', 'F']),
        }
        other_annotations = sum(1 for s in beat_types if s not in ['N', 'A', 'a', 'J', 'S', 'e', 'j', 'V', 'E', 'F'])
        
        summaries.append({
            'Record ID': record_id,
            'Sampling Frequency (Hz)': fs,
            'Num Samples': num_samples,
            'Duration (min)': round(duration_min, 2),
            'Num Annotations': num_annotations,
            'Normal Beats': class_counts['Normal'],
            'Supraventricular and Ventricular Beats': class_counts['SupraventricularAndVentricular'],
            'Other Annotations': other_annotations
        })
    
    df = pd.DataFrame(summaries)
    return df


if __name__ == "__main__":
    data_dir = 'data/mitdb'  
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist. Please download the MIT-BIH dataset.")
    else:
        summary_df = get_dataset_summary(data_dir)
        
        print("\nMIT-BIH Arrhythmia Database Summary:")
        print(summary_df)
        
        # Save to CSV (optional)
        summary_df.to_csv('mit_bih_summary.csv', index=False)
        print("\nSummary saved to 'mit_bih_summary.csv'")
        
        # Example: Load and print details for record 100
        signal, rpeaks, fs, ann = load_ecg('100', data_dir)
        if signal is not None:
            print(f"\nDetails for record 100:")
            print(f"Sampling Frequency: {fs} Hz")
            print(f"Number of Samples: {len(signal)}")
            print(f"Duration: {len(signal)/fs/60:.2f} minutes")
            print(f"Number of R-peaks: {len(rpeaks)}")
            print(f"First few R-peak indices: {rpeaks[:5]}")

