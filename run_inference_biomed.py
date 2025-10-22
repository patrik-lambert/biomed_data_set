#!/usr/bin/env python3
"""
Script to run inference on JSONL data using llama.cpp.

This script reads a JSONL file, extracts prompts from each line,
runs inference with llama.cpp, and outputs the results.
"""

import json
import subprocess
import argparse
import sys
import os
import tempfile
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on plain text data using llama.cpp"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input plain text file"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the llama.cpp model file"
    )
    parser.add_argument(
        "--output",
        help="Path to output file (default: stdout)"
    )
    parser.add_argument(
        "--llama-cpp-path",
        default="llama-cli",
        help="Path to llama-cli executable (default: llama-cli)"
    )    
    return parser.parse_args()


def extract_prompt(data_item: Dict[str, Any]) -> str:
    """Extract prompt from plain text data item."""
    return data_item.strip() + '\n'


def run_llama_inference(
    prompt: str,
    model_path: str,
    llama_cpp_path: str,
    gpu_layers: int,
    context_len: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    verbose: bool = False
) -> str:
    """Run inference using llama.cpp."""
    
    # Create a temporary file for the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(prompt)
        temp_file_path = temp_file.name
    
    try:
        # Prepare llama.cpp command
        # llama-cli -m Qwen-Qwen3-0.6B-f32.gguf -ngl 999 -c 2048 -s 42 --temp 0 --top-p 1.0 --top-k 0 -p "Hello" -fa off
        cmd = [
            llama_cpp_path,
            "-m", model_path,
            "-ngl", str(gpu_layers),
            "-c", str(context_len),
            "-s", str(seed),
            "-f", temp_file_path,
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "-fa", "off",
            "-n", "50", # number of tokens to generate
            "--no-display-prompt"
        ]
        
        if verbose:
            print(f"Running command: {' '.join(cmd)}", file=sys.stderr)
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Error running llama.cpp: {result.stderr}", file=sys.stderr)
            return ""
        
        # Extract the generated text (remove the prompt from output)
        output = result.stdout.strip()
        
        # Remove the original prompt from the output if it appears
        if prompt in output:
            output = output.replace(prompt, "").strip()
        
        return output
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running inference for prompt: {prompt[:50]}...", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Error running inference: {e}", file=sys.stderr)
        return ""
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass


def format_output(line: str, generated_text: str, time_inference: datetime.timedelta) -> Dict[str, Any]:
    """Format the output with original data and generated text."""
    # Split generated_text at the first newline character
    if "\n" in generated_text:
        generated_text = generated_text.split('\n', 1)[0]
    output = {
        "prompt": line,
        "generated_text": generated_text,
        "inference_time": time_inference.total_seconds()
    }
    return output


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found", file=sys.stderr)
        #sys.exit(1)
    
    # Validate llama-cpp executable
    try:
        subprocess.run([args.llama_cpp_path, "--help"], capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"Error: Cannot find or run '{args.llama_cpp_path}'", file=sys.stderr)
        print("Make sure llama.cpp is installed and the path is correct", file=sys.stderr)
        #sys.exit(1)
    
    # Read input data
    print(f"Reading data from {args.input}", file=sys.stderr)
    
    # Prepare output
    output_file = None
    if args.output:
        output_file = open(args.output, 'w', encoding='utf-8')
    
    try:
        # Process each segment
        with open(args.input, 'r', encoding='utf-8') as fin:
            for i, line in enumerate(fin):
                print(f"Processing segment {i+1}", file=sys.stderr)
                
                try:
                    # Extract prompt
                    prompt = line.strip()
                    # INSERT_YOUR_CODE
                    time_start = datetime.datetime.now()
                    # Run inference
                    generated_text = run_llama_inference(
                        prompt=prompt,
                        model_path=args.model,
                        llama_cpp_path=args.llama_cpp_path,
                        gpu_layers=999,
                        context_len=2048,
                        seed=42,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                        verbose=True
                    )
                    time_inference = datetime.datetime.now()-time_start
                    print(f"Time taken: {time_inference}", file=sys.stderr)
                    
                    # Format output
                    output = format_output(line, generated_text, time_inference)
                    
                    # Write output
                    output_line = json.dumps(output, ensure_ascii=False)
                    if output_file:
                        output_file.write(output_line + '\n')
                    else:
                        print(output_line)
                    
                except Exception as e:
                    print(f"Error processing item {i+1}: {e}", file=sys.stderr)
                    continue
    
    finally:
        if output_file:
            output_file.close()
    
    print("Processing complete", file=sys.stderr)


if __name__ == "__main__":
    main()
