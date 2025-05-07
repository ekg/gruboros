#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq

def find_parquet_files(paths):
    """Find all parquet files in the given paths."""
    parquet_files = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() == '.parquet':
            parquet_files.append(path)
        elif path.is_dir():
            parquet_files.extend(list(path.glob('**/*.parquet')))
    return parquet_files

def process_file(file_path, min_length=0, output_file=sys.stdout):
    """Process a single parquet file, extracting text fields."""
    try:
        # Read the parquet file using pyarrow
        table = pq.read_table(file_path, columns=['text'])
        
        # Convert to a pandas DataFrame for easier processing
        df = table.to_pandas()
        
        # Filter by minimum length if specified
        if min_length > 0:
            df = df[df['text'].str.len() >= min_length]
        
        # Write each text with null delimiter
        for text in df['text']:
            if isinstance(text, str):
                output_file.write(text + '\0')
                
        return len(df)
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return 0

def main():
    parser = argparse.ArgumentParser(description='Extract text fields from parquet files and concatenate with null delimiters.')
    parser.add_argument('paths', nargs='+', help='Directories or files to process')
    parser.add_argument('--min-length', type=int, default=0, help='Minimum length of text to include')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')
    args = parser.parse_args()
    
    # Find all parquet files
    parquet_files = find_parquet_files(args.paths)
    
    if not parquet_files:
        print("No parquet files found in the specified paths.", file=sys.stderr)
        return 1
    
    print(f"Found {len(parquet_files)} parquet files to process.", file=sys.stderr)
    
    # Setup output
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
    else:
        output_file = sys.stdout
    
    try:
        # Process each file
        total_texts = 0
        for i, file_path in enumerate(parquet_files, 1):
            print(f"Processing file {i}/{len(parquet_files)}: {file_path}", file=sys.stderr)
            texts_processed = process_file(file_path, args.min_length, output_file)
            total_texts += texts_processed
            print(f"  Processed {texts_processed} texts", file=sys.stderr)
        
        print(f"Total texts processed: {total_texts}", file=sys.stderr)
    finally:
        # Close output file if not stdout
        if args.output:
            output_file.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())