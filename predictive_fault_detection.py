"""
Adapt CARLA for Predictive Fault Detection

This script modifies the data preparation to enable prediction of faults
BEFORE they occur by:
1. Shifting fault labels forward in time (predict N timesteps ahead)
2. Training CARLA to detect precursor patterns that lead to faults
3. Using anomaly scores as early warning indicators
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path

def create_predictive_labels(labeled_anomalies_file, output_file, prediction_horizon=100):
    """
    Create predictive labels by shifting fault labels forward in time.
    
    Args:
        labeled_anomalies_file: Path to original labeled_anomalies.csv
        output_file: Path to save modified labeled_anomalies.csv
        prediction_horizon: Number of timesteps to predict ahead (default: 100)
                            This means: detect fault patterns N timesteps before they occur
    
    Returns:
        Modified DataFrame with shifted anomaly sequences
    """
    df = pd.read_csv(labeled_anomalies_file)
    
    print(f"Creating predictive labels with {prediction_horizon} timestep prediction horizon...")
    print("="*80)
    
    modified_sequences = []
    
    for idx, row in df.iterrows():
        sequences = ast.literal_eval(row['anomaly_sequences'])
        num_values = row['num_values']
        
        # Shift each anomaly sequence backward by prediction_horizon
        # This makes the model learn to detect faults N timesteps before they occur
        predictive_sequences = []
        
        for seq in sequences:
            start, end = seq
            
            # Shift backward: if fault occurs at timestep 1000, 
            # we want to detect it at timestep 900 (if horizon=100)
            new_start = max(0, start - prediction_horizon)
            new_end = max(0, end - prediction_horizon)
            
            # Only keep if the shifted sequence is valid
            if new_end > 0 and new_start < num_values:
                predictive_sequences.append([new_start, new_end])
        
        if len(predictive_sequences) > 0:
            modified_sequences.append(predictive_sequences)
            print(f"{row['chan_id']:8s}: {len(sequences):3d} original sequences -> "
                  f"{len(predictive_sequences):3d} predictive sequences")
        else:
            modified_sequences.append([])
            print(f"{row['chan_id']:8s}: {len(sequences):3d} original sequences -> "
                  f"0 predictive sequences (all shifted out of range)")
    
    # Update the DataFrame
    df['anomaly_sequences'] = [str(seq) for seq in modified_sequences]
    
    # Save modified file
    df.to_csv(output_file, index=False)
    print(f"\nSaved predictive labels to: {output_file}")
    print(f"\nNote: Model will now learn to detect faults {prediction_horizon} timesteps BEFORE they occur!")
    
    return df


def analyze_precursor_patterns(csv_file, npy_file, labeled_anomalies_file, 
                                prediction_horizon=100, window_size=200):
    """
    Analyze patterns that precede faults to understand what the model should learn.
    
    Args:
        csv_file: Original CSV with fault labels
        npy_file: Path to test data .npy file
        labeled_anomalies_file: Path to labeled_anomalies.csv
        prediction_horizon: How many timesteps before fault to analyze
        window_size: Window size used by CARLA
    """
    print("\n" + "="*80)
    print("Analyzing Precursor Patterns (patterns before faults)")
    print("="*80)
    
    # Load data
    data = np.load(npy_file)
    df_anomalies = pd.read_csv(labeled_anomalies_file)
    
    # Get anomaly sequences
    row = df_anomalies.iloc[0]  # Assuming Custom-1
    sequences = ast.literal_eval(row['anomaly_sequences'])
    
    # Analyze patterns before each fault
    precursor_windows = []
    fault_windows = []
    
    for seq in sequences[:5]:  # Analyze first 5 faults
        start, end = seq
        
        # Get data window BEFORE the fault (precursor)
        precursor_start = max(0, start - prediction_horizon - window_size)
        precursor_end = max(0, start - prediction_horizon)
        
        if precursor_end > precursor_start:
            precursor_data = data[precursor_start:precursor_end]
            precursor_windows.append(precursor_data)
            
            # Get data during the fault
            fault_data = data[start:min(end+1, len(data))]
            fault_windows.append(fault_data)
            
            print(f"\nFault at timesteps [{start}, {end}]:")
            print(f"  Precursor window: [{precursor_start}, {precursor_end}] "
                  f"({precursor_end - precursor_start} timesteps before fault)")
            print(f"  Fault window: [{start}, {end}] "
                  f"({end - start + 1} timesteps)")
    
    if len(precursor_windows) > 0:
        print(f"\n✓ Found {len(precursor_windows)} precursor patterns to learn")
        print("  CARLA will learn to detect these patterns as early warnings")
    
    return precursor_windows, fault_windows


def create_predictive_config(base_config_file, output_config_file, prediction_horizon=100):
    """
    Create a modified config file for predictive fault detection.
    Adds notes about prediction horizon.
    """
    with open(base_config_file, 'r') as f:
        config_content = f.read()
    
    # Add comment about prediction
    header = f"# Predictive Fault Detection Configuration\n"
    header += f"# Prediction Horizon: {prediction_horizon} timesteps\n"
    header += f"# Model will detect faults {prediction_horizon} timesteps before they occur\n"
    header += f"# Original config:\n\n"
    
    with open(output_config_file, 'w') as f:
        f.write(header + config_content)
    
    print(f"Created predictive config: {output_config_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Adapt CARLA for Predictive Fault Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict faults 100 timesteps ahead (default)
  python predictive_fault_detection.py --horizon 100
  
  # Predict faults 200 timesteps ahead (more early warning)
  python predictive_fault_detection.py --horizon 200
  
  # Predict faults 50 timesteps ahead (less early warning, more accurate)
  python predictive_fault_detection.py --horizon 50
        """
    )
    
    parser.add_argument('--labeled_anomalies', type=str, 
                       default='datasets/MSL_SMAP/labeled_anomalies.csv',
                       help='Path to labeled_anomalies.csv')
    parser.add_argument('--output', type=str,
                       default='datasets/MSL_SMAP/labeled_anomalies_predictive.csv',
                       help='Output path for predictive labels')
    parser.add_argument('--horizon', type=int, default=100,
                       help='Prediction horizon: how many timesteps ahead to predict (default: 100)')
    parser.add_argument('--test_data', type=str,
                       default='datasets/MSL_SMAP/test/Custom-1.npy',
                       help='Path to test data .npy file for analysis')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze precursor patterns before faults')
    
    args = parser.parse_args()
    
    # Create predictive labels
    df = create_predictive_labels(
        args.labeled_anomalies,
        args.output,
        args.horizon
    )
    
    # Analyze precursor patterns if requested
    if args.analyze and Path(args.test_data).exists():
        analyze_precursor_patterns(
            None,  # CSV not needed for this analysis
            args.test_data,
            args.labeled_anomalies,
            args.horizon
        )
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print(f"1. Use the modified labeled_anomalies file: {args.output}")
    print(f"2. Train CARLA as usual - it will now learn to predict faults {args.horizon} timesteps ahead")
    print(f"3. When you get anomaly scores, they indicate early warning signs")
    print(f"4. High scores = fault likely to occur in next {args.horizon} timesteps")
    print("\nNote: You may need to adjust prediction_horizon based on your data:")
    print("  - Too large: May detect too early (more false positives)")
    print("  - Too small: May detect too late (less predictive value)")
    print("  - Recommended: Start with 50-200 timesteps, tune based on results")

