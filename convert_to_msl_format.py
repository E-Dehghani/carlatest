"""
Convert LabeledData_2022.csv to MSL dataset format
Creates train/test .npy files and labeled_anomalies.csv
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def find_anomaly_sequences(labels):
    """
    Find contiguous anomaly regions in the label array.
    Returns list of [start, end] tuples (inclusive).
    """
    anomalies = []
    in_anomaly = False
    start_idx = None
    
    for i, label in enumerate(labels):
        if label == 1 and not in_anomaly:
            # Start of anomaly region
            start_idx = i
            in_anomaly = True
        elif label == 0 and in_anomaly:
            # End of anomaly region
            anomalies.append([start_idx, i - 1])
            in_anomaly = False
    
    # Handle case where anomaly extends to end of data
    if in_anomaly:
        anomalies.append([start_idx, len(labels) - 1])
    
    return anomalies

def convert_to_msl_format(csv_file, output_dir, channel_name='Custom-1', spacecraft='CUSTOM', num_features=None, max_rows=None):
    """
    Convert CSV file to MSL format.
    
    Args:
        csv_file: Path to LabeledData_2022.csv
        output_dir: Directory where train/test folders will be created
        channel_name: Name for the channel (e.g., 'Custom-1')
        spacecraft: Name for spacecraft (e.g., 'CUSTOM')
        num_features: Number of feature columns to keep. If None, keeps last 3 by default.
                     Can be an integer (keeps last N columns) or a list of column names/indices.
        max_rows: Maximum number of rows to process. If None, processes all rows.
                 Useful for reducing dataset size for testing or memory constraints.
    """
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Limit rows if max_rows is specified
    original_rows = len(df)
    if max_rows is not None and max_rows > 0:
        if max_rows < original_rows:
            df = df.iloc[:max_rows].copy()
            print(f"Limited dataset from {original_rows:,} rows to {max_rows:,} rows")
        else:
            print(f"max_rows ({max_rows:,}) is >= total rows ({original_rows:,}). Using all rows.")
    else:
        print(f"Processing all {original_rows:,} rows")
    
    # Get all columns
    all_columns = df.columns.tolist()
    date_column = all_columns[0]  # First column is Date
    class_column = all_columns[-1]  # Last column is 'class'
    
    # Get feature columns (all columns except first 'Date' and last 'class')
    all_feature_columns = all_columns[1:-1]
    
    print(f"Data shape: {df.shape}")
    print(f"Total feature columns available: {len(all_feature_columns)}")
    print(f"All feature columns: {all_feature_columns}")
    
    # Select which feature columns to keep
    if num_features is None:
        # Default: keep last 3 columns
        num_features = 3
        data_columns = all_feature_columns[-3:]
        print(f"\nUsing default: keeping last {num_features} feature columns")
    elif isinstance(num_features, int):
        # Keep last N columns
        if num_features > len(all_feature_columns):
            print(f"Warning: Requested {num_features} columns but only {len(all_feature_columns)} available. Using all.")
            num_features = len(all_feature_columns)
        data_columns = all_feature_columns[-num_features:]
        print(f"\nKeeping last {num_features} feature columns")
    elif isinstance(num_features, list):
        # User specified column names or indices
        data_columns = []
        for col in num_features:
            if isinstance(col, int):
                # Index-based selection
                if 0 <= col < len(all_feature_columns):
                    data_columns.append(all_feature_columns[col])
                else:
                    print(f"Warning: Column index {col} out of range. Skipping.")
            elif isinstance(col, str):
                # Name-based selection
                if col in all_feature_columns:
                    data_columns.append(col)
                else:
                    print(f"Warning: Column '{col}' not found. Skipping.")
        if len(data_columns) == 0:
            raise ValueError("No valid columns selected!")
        print(f"\nUsing {len(data_columns)} specified columns")
    else:
        raise ValueError(f"num_features must be None, int, or list, got {type(num_features)}")
    
    print(f"Selected feature columns: {data_columns}")
    print(f"Number of features: {len(data_columns)}")
    
    # Extract data and labels
    data = df[data_columns].values.astype(np.float64)
    labels_raw = df[class_column].values
    
    # Convert labels: 0 = normal (0), anything else = anomaly (1)
    # User specified: class 0 is normal, class 1 is anomaly
    # Treat any non-zero as anomaly
    labels = (labels_raw != 0).astype(int)
    
    # Handle NaN values
    if np.any(np.isnan(data)):
        print("Warning: Data contains NaN values. Replacing with zeros.")
        data = np.nan_to_num(data)
    
    # Time series split: 70% train, 30% test (not random!)
    total_rows = len(data)
    train_size = int(total_rows * 0.7)
    
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    
    test_data = data[train_size:]
    test_labels = labels[train_size:]
    
    print(f"\nSplit information:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Train rows: {len(train_data):,} ({len(train_data)/total_rows*100:.1f}%)")
    print(f"  Test rows: {len(test_data):,} ({len(test_data)/total_rows*100:.1f}%)")
    print(f"  Train anomalies: {np.sum(train_labels):,} ({np.sum(train_labels)/len(train_labels)*100:.2f}%)")
    print(f"  Test anomalies: {np.sum(test_labels):,} ({np.sum(test_labels)/len(test_labels)*100:.2f}%)")
    
    # Create output directories
    train_dir = Path(output_dir) / 'train'
    test_dir = Path(output_dir) / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train data (no labels needed for train in MSL format)
    train_file = train_dir / f'{channel_name}.npy'
    np.save(train_file, train_data)
    print(f"\nSaved train data: {train_file}")
    print(f"  Shape: {train_data.shape} (timesteps, features)")
    
    # Save test data
    test_file = test_dir / f'{channel_name}.npy'
    np.save(test_file, test_data)
    print(f"Saved test data: {test_file}")
    print(f"  Shape: {test_data.shape} (timesteps, features)")
    
    # Find anomaly sequences in test data
    anomaly_sequences = find_anomaly_sequences(test_labels)
    
    print(f"\nAnomaly sequences in test data: {len(anomaly_sequences)}")
    if len(anomaly_sequences) > 0:
        print(f"  First few: {anomaly_sequences[:5]}")
    
    # Create labeled_anomalies.csv entry
    # Format: chan_id, spacecraft, anomaly_sequences, class, num_values
    labeled_anomalies_file = Path(output_dir) / 'labeled_anomalies.csv'
    
    # Determine anomaly class type (simplified - using 'point' for now)
    # You can enhance this to detect contextual vs point anomalies
    anomaly_class = '[point]' if len(anomaly_sequences) <= 3 else '[contextual]'
    
    # Create DataFrame for labeled_anomalies.csv
    labeled_anomalies_df = pd.DataFrame({
        'chan_id': [channel_name],
        'spacecraft': [spacecraft],
        'anomaly_sequences': [str(anomaly_sequences)],
        'class': [anomaly_class],
        'num_values': [len(test_data)]
    })
    
    # Check if labeled_anomalies.csv already exists
    # Always create new file (overwrite if exists)
    if labeled_anomalies_file.exists():
        print(f"\nOverwriting existing labeled_anomalies.csv")
    else:
        print(f"\nCreating new labeled_anomalies.csv")
    
    labeled_anomalies_df.to_csv(labeled_anomalies_file, index=False)
    
    labeled_anomalies_df.to_csv(labeled_anomalies_file, index=False)
    print(f"Saved labeled_anomalies.csv: {labeled_anomalies_file}")
    
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    train/{channel_name}.npy")
    print(f"    test/{channel_name}.npy")
    print(f"    labeled_anomalies.csv")
    print(f"\nYou can now use this dataset with MSL format.")
    print(f"Set the dataset root path in utils/mypath.py or config files.")
    print(f"\nNote: Using {len(data_columns)} features. Update 'in_channels' in config to {len(data_columns)}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSV to MSL format')
    parser.add_argument('--csv_file', type=str, default='LabeledData_2022.csv',
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='datasets/MSL_SMAP',
                        help='Output directory for train/test folders')
    parser.add_argument('--channel_name', type=str, default='Custom-1',
                        help='Channel identifier (e.g., Custom-1)')
    parser.add_argument('--spacecraft', type=str, default='CUSTOM',
                        help='Spacecraft name (e.g., CUSTOM)')
    parser.add_argument('--num_features', type=int, default=None,
                        help='Number of feature columns to keep (keeps last N columns). Default: 3')
    parser.add_argument('--feature_columns', type=str, nargs='+', default=None,
                        help='Specific column names to keep (space-separated). Overrides --num_features if provided.')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='Maximum number of rows to process. If not specified, processes all rows. Useful for reducing dataset size.')
    
    args = parser.parse_args()
    
    # Handle feature column selection
    if args.feature_columns:
        # User specified column names
        num_features = args.feature_columns
    else:
        # Use num_features (None means default 3)
        num_features = args.num_features
    
    convert_to_msl_format(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        channel_name=args.channel_name,
        spacecraft=args.spacecraft,
        num_features=num_features,
        max_rows=args.max_rows
    )