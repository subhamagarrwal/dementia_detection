import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import gc
warnings.filterwarnings('ignore')

class SleepWakeFeatureExtractor:
    """Extract sleep/wake features from accelerometer data"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.sampling_rate = 100  # Hz - typical for wrist accelerometers
        self.window_size = 30  # seconds
        self.window_samples = self.sampling_rate * self.window_size  # 3000 samples
        
    def load_participant_data(self, pid):
        """Load participant accelerometer data"""
        filepath = self.data_dir / f'{pid}.csv'
        print(f"Loading {pid}...")
        
        # Load in chunks to handle large files
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=50000, low_memory=False):
            chunks.append(chunk)
        
        data = pd.concat(chunks, ignore_index=True)
        print(f"  Loaded {len(data):,} rows")
        
        return data
    
    def calculate_magnitude(self, x, y, z):
        """Calculate magnitude of 3D accelerometer vector"""
        return np.sqrt(x**2 + y**2 + z**2)
    
    def extract_window_features(self, window_data):
        """Extract features from a 30-second window"""
        
        # Calculate magnitude if x, y, z columns exist
        if all(col in window_data.columns for col in ['x', 'y', 'z']):
            magnitude = self.calculate_magnitude(
                window_data['x'].values,
                window_data['y'].values,
                window_data['z'].values
            )
        else:
            # Use first numeric column if x,y,z not available
            numeric_cols = window_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return None
            magnitude = window_data[numeric_cols[0]].values
        
        # Feature 1: Variance
        variance = np.var(magnitude)
        
        # Feature 2: Mean
        mean = np.mean(magnitude)
        
        # Feature 3: Zero Crossing Rate
        # Count how many times signal crosses its mean
        centered = magnitude - mean
        zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
        zero_crossing_rate = zero_crossings / len(magnitude)
        
        # Additional useful features
        std = np.std(magnitude)
        range_val = np.max(magnitude) - np.min(magnitude)
        
        return {
            'variance': variance,
            'mean': mean,
            'std': std,
            'zero_crossing_rate': zero_crossing_rate,
            'range': range_val
        }
    
    def create_sleep_labels_simple(self, data, sleep_start_hour=22, wake_hour=7):
        """
        Create simple sleep labels based on time of day
        Assumes sleep period: 10 PM to 7 AM
        
        For real implementation, you'd use actual sleep diary data
        """
        if 'time' in data.columns:
            time_col = 'time'
        elif 'timestamp' in data.columns:
            time_col = 'timestamp'
        else:
            # No time column, use simple heuristic based on row index
            print("  Warning: No time column found, using simple heuristic")
            return None
        
        # Parse timestamps
        data[time_col] = pd.to_datetime(data[time_col])
        
        # Extract hour
        data['hour'] = data[time_col].dt.hour
        
        # Label as sleep (1) if between sleep_start_hour and wake_hour
        # Handle wraparound (22:00 to 07:00)
        is_sleep = ((data['hour'] >= sleep_start_hour) | (data['hour'] < wake_hour))
        
        return is_sleep.astype(int)
    
    def process_participant(self, pid, output_file):
        """Process one participant and extract features"""
        
        print(f"\n{'='*60}")
        print(f"Processing {pid}")
        print(f"{'='*60}")
        
        try:
            # Load data
            data = self.load_participant_data(pid)
            
            # Check columns
            print(f"  Columns: {list(data.columns)}")
            
            # Create sleep labels (simplified - using time-based heuristic)
            sleep_labels = self.create_sleep_labels_simple(data)
            
            if sleep_labels is None:
                print("  Skipping - no time column")
                return 0
            
            # Calculate number of windows
            total_windows = len(data) // self.window_samples
            print(f"  Creating {total_windows} windows of {self.window_size} seconds")
            
            # Process windows
            features_list = []
            
            for i in range(total_windows):
                start_idx = i * self.window_samples
                end_idx = start_idx + self.window_samples
                
                # Get window data
                window = data.iloc[start_idx:end_idx]
                
                # Extract features
                features = self.extract_window_features(window)
                
                if features is None:
                    continue
                
                # Get majority sleep label for this window
                window_sleep_label = sleep_labels.iloc[start_idx:end_idx].mode()[0]
                
                # Add label
                features['is_sleep'] = window_sleep_label
                features['participant'] = pid
                features['window_index'] = i
                
                features_list.append(features)
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"    Processed {i+1}/{total_windows} windows...", end='\r')
            
            print(f"\n  Extracted {len(features_list)} feature windows")
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Append to output file
            if not Path(output_file).exists():
                # Write with header if file doesn't exist
                features_df.to_csv(output_file, index=False, mode='w')
                print(f"  Created new file: {output_file}")
            else:
                # Append without header if file exists
                features_df.to_csv(output_file, index=False, mode='a', header=False)
                print(f"  Appended to: {output_file}")
            
            # Clean up memory
            del data
            del features_df
            gc.collect()
            
            return len(features_list)
            
        except Exception as e:
            print(f"  ERROR processing {pid}: {str(e)}")
            return 0
    
    def process_all_participants(self, start_id=1, end_id=87, output_file=None):
        """Process all participants from start_id to end_id"""
        
        print("\n" + "="*60)
        print(f"PROCESSING ALL PARTICIPANTS: P{start_id:03d} to P{end_id:03d}")
        print("="*60)
        
        total_features = 0
        successful = 0
        failed = 0
        
        for i in tqdm(range(start_id, end_id + 1), desc="Processing participants"):
            pid = f'P{i:03d}'
            
            num_features = self.process_participant(pid, output_file)
            
            if num_features > 0:
                total_features += num_features
                successful += 1
            else:
                failed += 1
        
        return {
            'total_features': total_features,
            'successful': successful,
            'failed': failed
        }


# Main execution
if __name__ == "__main__":
    
    # Setup paths
    data_dir = r"c:\Users\subha\Desktop\Projects\dementia_detection\data\capture24_dataset"
    output_file = r"c:\Users\subha\Desktop\Projects\dementia_detection\data\sleep_training_data.csv"
    
    # Initialize extractor
    extractor = SleepWakeFeatureExtractor(data_dir)
    
    print("\n" + "="*60)
    print("PHASE 1: SLEEP/WAKE FEATURE EXTRACTION - ALL PARTICIPANTS")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    print(f"Output file: {output_file}")
    
    # Delete existing output file to start fresh
    if Path(output_file).exists():
        response = input(f"\n{output_file} already exists. Overwrite? (y/n): ")
        if response.lower() == 'y':
            Path(output_file).unlink()
            print("Existing file deleted.")
        else:
            print("Appending to existing file.")
    
    # Process ALL participants from P001 to P087
    stats = extractor.process_all_participants(
        start_id=1, 
        end_id=87, 
        output_file=output_file
    )
    
    # Load and display final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if Path(output_file).exists():
        results = pd.read_csv(output_file)
        
        print(f"\nTotal feature windows extracted: {len(results):,}")
        print(f"Participants successfully processed: {stats['successful']}")
        print(f"Participants failed: {stats['failed']}")
        
        print(f"\nColumns: {list(results.columns)}")
        
        print(f"\nFirst 10 rows:")
        print(results.head(10))
        
        print(f"\nLast 10 rows:")
        print(results.tail(10))
        
        print(f"\nSleep/Wake distribution:")
        print(results['is_sleep'].value_counts())
        print(f"\nOverall sleep percentage: {results['is_sleep'].mean()*100:.1f}%")
        
        print(f"\nFeatures per participant:")
        print(results.groupby('participant').size().describe())
        
        print(f"\nFeature statistics:")
        print(results[['variance', 'mean', 'std', 'zero_crossing_rate', 'range']].describe())
        
        # Save summary
        summary_file = output_file.replace('.csv', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("SLEEP/WAKE FEATURE EXTRACTION SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total windows: {len(results):,}\n")
            f.write(f"Participants processed: {stats['successful']}\n")
            f.write(f"Sleep percentage: {results['is_sleep'].mean()*100:.1f}%\n\n")
            f.write("Participant distribution:\n")
            f.write(results.groupby('participant').size().to_string())
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
    print("\n" + "="*60)
    print("✓ Phase 1 complete for ALL participants!")
    print("="*60)
    print(f"\nOutput file: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Review the output file")
    print(f"2. Verify features look reasonable")
    print(f"3. Train sleep/wake classifier using this data")
    print(f"4. Save trained model as sleep_wake_model.pkl")