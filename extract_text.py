#!/usr/bin/env python3
"""
Script to extract text content from JSONL files.

This script reads a JSONL file, extracts the content of the "text" field
from each line, and outputs it as plain text.
"""

import json
import argparse
import sys
from pathlib import Path


def extract_text_from_jsonl(input_file: str, output_file: str = None):
    """
    Extract text content from JSONL file and output as plain text.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to output file (optional, defaults to stdout)
    """
    try:
        # Open input file
        with open(input_file, 'r', encoding='utf-8') as f:
            # Open output file or use stdout
            if output_file:
                out_f = open(output_file, 'w', encoding='utf-8')
            else:
                out_f = sys.stdout
            
            try:
                line_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON line
                        data = json.loads(line)
                        
                        # Extract text field
                        if 'text' in data:
                            text_content = data['text']
                            # Preserve literal \n characters instead of converting to newlines
                            text_content = text_content.replace('\n', '\\n')
                            out_f.write(text_content + '\n')
                            line_count += 1
                        else:
                            print(f"Warning: No 'text' field found on line {line_num}", file=sys.stderr)
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}", file=sys.stderr)
                        continue
                
                print(f"Extracted text from {line_count} lines", file=sys.stderr)
                
            finally:
                if output_file:
                    out_f.close()
                    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract text content from JSONL files"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Extract text
    extract_text_from_jsonl(args.input_file, args.output)


if __name__ == "__main__":
    main()
