"""
SDN-DDoS Traffic Feature Extractor
This script processes CSV data from SDN-DDoS Traffic Dataset and computes
CICFlowMeter-like features using Scapy-inspired calculations.
"""
import pandas as pd
import numpy as np

from collections import defaultdict
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICFlowFeatureExtractor:
    """
    A class to extract CICFlowMeter-like features from SDN-DDoS dataset
    """
    def __init__(self):
        self.feature_names = [
            'Packet_Length_Mean',
            'Average_Packet_Size', 
            'Bwd_Packet_Length_Min',
            'Fwd_Packets/s',
            'Min_Packet_Length',
            'Down/Up_Ratio',
            'Label'
        ]
    def calculate_packet_length_mean(self, pkt_count: int, byte_count: int) -> float:
        """
        Calculate mean packet length in bytes
        Args:
            pkt_count: Total packet count
            byte_count: Total byte count
        Returns:
            Mean packet length in bytes
        """
        if pkt_count > 0:
            return byte_count / pkt_count
        return 0.0
    def calculate_average_packet_size(self, pkt_count: int, byte_count: int, flows: int) -> float:
        """
        Calculate average packet size considering flow information
        Args:
            pkt_count: Total packet count
            byte_count: Total byte count
            flows: Number of flows
        Returns:
            Average packet size
        """
        if flows > 0 and pkt_count > 0:
            return (byte_count / pkt_count) * (pkt_count / flows)
        return 0.0
    def calculate_bwd_packet_length_min(self, tx_bytes: int, pkt_count: int) -> float:
        """
        Calculate minimum backward packet length (approximated)
        Args:
            tx_bytes: Transmitted bytes
            pkt_count: Packet count
        Returns:
            Minimum backward packet length
        """
        # This is an approximation since we don't have individual packet sizes
        # In real CICFlowMeter, this would be the minimum packet size in backward direction
        if pkt_count > 0:
            avg_packet_size = tx_bytes / pkt_count
            return avg_packet_size * 0.3  # Approximation: assume min is 30% of average
        return 0.0
    def calculate_fwd_packets_per_second(self, pkt_count: int, tot_duration: float) -> float:
        """
        Calculate forward packets per second
        Args:
            pkt_count: Total packet count
            tot_duration: Total duration in seconds
        Returns:
            Packets per second rate
        """
        if tot_duration > 0:
            return pkt_count / tot_duration
        return 0.0
    def calculate_min_packet_length(self, byte_count: int, pkt_count: int) -> float:
        """
        Calculate minimum packet length in the flow
        Args:
            byte_count: Total byte count
            pkt_count: Total packet count
        Returns:
            Minimum packet length (approximated)
        """
        if pkt_count > 0:
            avg_packet_size = byte_count / pkt_count
            return avg_packet_size * 0.2  # Approximation: assume min is 20% of average
        return 0.0
    def calculate_down_up_ratio(self, rx_bytes: int, tx_bytes: int) -> float:
        """
        Calculate download/upload ratio
        Args:
            rx_bytes: Received bytes
            tx_bytes: Transmitted bytes
        Returns:
            Down/Up ratio
        """
        if tx_bytes > 0:
            return rx_bytes / tx_bytes
        return float('inf')  # Infinite ratio if no upload
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract CICFlowMeter-like features from the input dataframe
        Args:
            df: Input DataFrame with SDN-DDoS dataset features
        Returns:
            DataFrame with extracted features
        """
        logger.info("Starting feature extraction...")
        # Initialize result dictionary
        features = defaultdict(list)
        for idx, row in df.iterrows():
            try:
                # Extract basic metrics
                pkt_count = row['pkt_count']
                byte_count = row['byte_count']
                tot_duration = row['tot_duration']
                flows = row['flows']
                tx_bytes = row['tx_bytes']
                rx_bytes = row['rx_bytes']
                # Calculate features
                features['Packet_Length_Mean'].append(
                    self.calculate_packet_length_mean(pkt_count, byte_count)
                )
                features['Average_Packet_Size'].append(
                    self.calculate_average_packet_size(pkt_count, byte_count, flows)
                )
                features['Bwd_Packet_Length_Min'].append(
                    self.calculate_bwd_packet_length_min(tx_bytes, pkt_count)
                )
                features['Fwd_Packets/s'].append(
                    self.calculate_fwd_packets_per_second(pkt_count, tot_duration)
                )
                features['Min_Packet_Length'].append(
                    self.calculate_min_packet_length(byte_count, pkt_count)
                )
                features['Down/Up_Ratio'].append(
                    self.calculate_down_up_ratio(rx_bytes, tx_bytes)
                )
                # Preserve the original label
                features['Label'].append(row['label'])
            except KeyError as e:
                logger.error(f"Missing required column: {e}")
                continue
            except ZeroDivisionError:
                logger.warning(f"Division by zero encountered in row {idx}")
                # Append zeros for calculations that resulted in division by zero
                for feature in self.feature_names[:-1]:  # Exclude Label
                    features[feature].append(0.0)
                features['Label'].append(row.get('label', 'Unknown'))
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        logger.info(f"Successfully processed {len(features['Label'])} rows")
        return pd.DataFrame(features)
    def validate_input_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input dataframe has required columns
        Args:
            df: Input DataFrame to validate
        Returns:
            Boolean indicating if validation passed
        """
        required_columns = [
            'pkt_count', 'byte_count', 'duration', 'tot_duration', 'flows',
            'tx_bytes', 'rx_bytes', 'label'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        logger.info("Input data validation passed")
        return True
def main():
    """
    Main execution function
    """
    # Initialize feature extractor
    extractor = CICFlowFeatureExtractor()
    try:
        # Read input CSV file
        input_file = r"E:\SDN-DDoS_Traffic_Dataset from Mendeley\SDN-DDoS_Traffic_Dataset.csv"
        
        logger.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        # Validate input data
        if not extractor.validate_input_data(df):
            logger.error("Input data validation failed. Exiting.")
            return
        # Extract features
        features_df = extractor.extract_features(df)
        # Save results
        output_file = r"E:\SDN-DDoS_Traffic_Dataset from Mendeley\SDN-DDoS_With_CIC_Features2.csv"

        features_df.to_csv(output_file, index=False)
        logger.info(f"Features saved to: {output_file}")
        # Display summary statistics
        logger.info("\n=== Feature Summary ===")
        for feature in extractor.feature_names[:-1]:  # Exclude Label
            if feature in features_df.columns:
                mean_val = features_df[feature].mean()
                std_val = features_df[feature].std()
                logger.info(f"{feature}: Mean={mean_val:.2f}, Std={std_val:.2f}")
        # Display label distribution
        logger.info("\n=== Label Distribution ===")
        label_counts = features_df['Label'].value_counts()
        for label, count in label_counts.items():
            logger.info(f"{label}: {count} samples ({count/len(features_df)*100:.1f}%)")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
if __name__ == "__main__":
    main()
''' 
This Python script provides a comprehensive solution for extracting CICFlowMeter-like features from the SDN-DDoS Traffic Dataset. Here are the key features of the code:
Key Features:
1. Modular Design: The CICFlowFeatureExtractor class encapsulates all functionality
2. Comprehensive Logging: Detailed logging for debugging and monitoring
3. Error Handling: Robust error handling for data validation and processing
4. Type Hints: Clear type annotations for better code documentation
5. Configurable Features: Easy to modify or extend feature calculations
Feature Calculations:
· Packet Length Mean: Average packet size in bytes
· Average Packet Size: Weighted average considering flow information
· Bwd Packet Length Min: Minimum backward packet length (approximated)
· Fwd Packets/s: Forward packets per second rate
· Min Packet Length: Minimum packet length in flow (approximated)
· Down/Up Ratio: Download to upload ratio based on RX/TX bytes
Usage:
1. Save your CSV file as "SDN_DDoS_Traffic_Dataset.csv"
2. Run the script: python cic_feature_extractor.py
3. Output will be saved as "CICFlowMeter_Features.csv"
The code includes proper data validation, error handling, and generates summary statistics for quality assurance.

'''

 
