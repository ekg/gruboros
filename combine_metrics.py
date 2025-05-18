#!/usr/bin/env python3
import os
import sys
import argparse
import re
import pandas as pd

def extract_model_info(summary_file):
    """Extract key information from model_summary.txt"""
    info = {}
    
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            
            # Extract model parameters
            param_match = re.search(r'Model parameters: ([\d,]+)', content)
            if param_match:
                # Remove commas and convert to integer
                info['parameters'] = int(param_match.group(1).replace(',', ''))
            
            # Extract model dimension
            dim_match = re.search(r'"dim": (\d+)', content)
            if dim_match:
                info['dimension'] = int(dim_match.group(1))
                
            # Extract model depth
            depth_match = re.search(r'"depth": (\d+)', content)
            if depth_match:
                info['depth'] = int(depth_match.group(1))
                
            # Extract sequence length
            seq_len_match = re.search(r'Sequence length: (\d+)', content)
            if seq_len_match:
                info['sequence_length'] = int(seq_len_match.group(1))
                
            # Extract batch size
            batch_match = re.search(r'Micro-batch size per GPU: (\d+)', content)
            if batch_match:
                info['batch_size'] = int(batch_match.group(1))
                
            # Extract gradient accumulation steps
            grad_accum_match = re.search(r'Gradient accumulation steps: (\d+)', content)
            if grad_accum_match:
                info['grad_accum'] = int(grad_accum_match.group(1))
                
            # Extract effective global batch size
            global_batch_match = re.search(r'Global batch size \(across all nodes, samples\): (\d+)', content)
            if global_batch_match:
                info['global_batch_size'] = int(global_batch_match.group(1))
                
            # Extract number of nodes
            nodes_match = re.search(r'Number of nodes \(data parallel replicas\): (\d+)', content)
            if nodes_match:
                info['num_nodes'] = int(nodes_match.group(1))
                
            # Extract training command (to parse for learning rate, etc.)
            cmd_match = re.search(r'Training command: (.*)', content)
            if cmd_match:
                command = cmd_match.group(1)
                
                # Extract learning rate
                lr_match = re.search(r'--lr (\d+\.\d+)', command)
                if lr_match:
                    info['learning_rate'] = float(lr_match.group(1))
                    
                # Extract weight decay
                wd_match = re.search(r'--weight_decay (\d+\.\d+)', command)
                if wd_match:
                    info['weight_decay'] = float(wd_match.group(1))
                    
                # Extract ScheduleFree beta
                sf_beta_match = re.search(r'--sf_beta (\d+\.\d+)', command)
                if sf_beta_match:
                    info['sf_beta'] = float(sf_beta_match.group(1))
                    
    except Exception as e:
        print(f"Error parsing {summary_file}: {e}", file=sys.stderr)
        
    return info

def process_directory(directory):
    """Process a single output directory and return a DataFrame with combined metrics"""
    # Path to model summary and metrics files
    summary_file = os.path.join(directory, 'model_summary.txt')
    metrics_file = os.path.join(directory, 'training_metrics.tsv')
    
    # Check if both files exist
    if not os.path.exists(summary_file):
        print(f"Warning: {summary_file} not found, skipping", file=sys.stderr)
        return None
        
    if not os.path.exists(metrics_file):
        print(f"Warning: {metrics_file} not found, skipping", file=sys.stderr)
        return None
    
    # Extract model information from summary file
    model_info = extract_model_info(summary_file)
    
    # Read training metrics
    try:
        metrics_df = pd.read_csv(metrics_file, sep='\t')
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}", file=sys.stderr)
        return None
    
    # Add model information as columns to metrics DataFrame
    for key, value in model_info.items():
        metrics_df[key] = value
        
    # Add directory name as model_name column
    metrics_df['model_name'] = os.path.basename(os.path.normpath(directory))
    
    # Add full directory path
    metrics_df['directory'] = os.path.abspath(directory)
    
    return metrics_df

def main():
    parser = argparse.ArgumentParser(description='Combine training metrics from multiple model runs')
    parser.add_argument('directories', nargs='+', help='Output directories to process')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Process each directory and collect DataFrames
    dfs = []
    for directory in args.directories:
        df = process_directory(directory)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print("Error: No valid data found in any of the specified directories", file=sys.stderr)
        sys.exit(1)
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Output combined data
    if args.output:
        combined_df.to_csv(args.output, sep='\t', index=False)
    else:
        # Print to stdout
        combined_df.to_csv(sys.stdout, sep='\t', index=False)

if __name__ == '__main__':
    main()
