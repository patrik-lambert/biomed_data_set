#!/usr/bin/env python3
"""
Script to run inference on JSONL data using llama.cpp.

This script reads a JSONL file, extracts prompts from each line,
runs inference with llama.cpp, and outputs the results.
"""

import subprocess
import argparse
import sys
import os
import tempfile
import datetime
from typing import Dict, Any


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
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (default: 0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling (default: 1.0)"
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling (default: 0.0)"
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty (default: 0.0)"
    )
    parser.add_argument(
        "--src-lang",
        required=True,
        help="Source language code (e.g., en, de, fr)"
    )
    parser.add_argument(
        "--tgt-lang",
        required=True,
        help="Target language code (e.g., en, de, fr)"
    )
    return parser.parse_args()


def build_prompt(model_name: str, src_lang: str, tgt_lang: str, text: str) -> str:
    """Build prompt for llama.cpp."""
    # Map common ISO 639-1 language codes to full language names
    lang_code_to_name = {
        "en": "English",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
        "cs": "Czech",
        "nl": "Dutch",
        "pl": "Polish",
        "sv": "Swedish",
        "fi": "Finnish",
        "tr": "Turkish",
        "el": "Greek",
        "no": "Norwegian",
        "da": "Danish",
        "he": "Hebrew",
        "vi": "Vietnamese",
        "uk": "Ukrainian",
        "bg": "Bulgarian",
        "hr": "Croatian",
        "th": "Thai",
        "id": "Indonesian",
        "ro": "Romanian",
        "hu": "Hungarian",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "et": "Estonian",
        "fa": "Persian",
        "sr": "Serbian",
        "sk": "Slovak",
        "sl": "Slovenian",
        # Add more as needed
    }
    src_lang_name = lang_code_to_name.get(src_lang.lower(), src_lang)
    tgt_lang_name = lang_code_to_name.get(tgt_lang.lower(), tgt_lang)
    # if "Qwen" in model_name:
    #     prompt = f"<|im_start|>user\nTranslate the following text from {src_lang_name} ({src_lang.upper()}) to {tgt_lang_name} ({tgt_lang.upper()}):\n{src_lang.upper()}: {text}\n{tgt_lang.upper()}:<|im_end|>\n<|im_start|>assistant\n/no_think"
    if "Qwen" in model_name:
        prompt = f"<|im_start|>user\nTranslate the following text from {src_lang_name} to {tgt_lang_name}:\n{text}\n<|im_end|>\n<|im_start|>assistant\n/no_think"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return prompt

def count_tokens(
    prompt: str,
    model_path: str,
    llama_cpp_path: str,
    verbose: bool = False
) -> int:
    """Count tokens using llama.cpp."""

    try:
        # Prepare llama.cpp command
        # llama-tokenize -m model_name -p 'Hello world.' --show-count
        llama_tokenize_bin = llama_cpp_path.replace("llama-cli", "llama-tokenize")
        cmd = [
            llama_tokenize_bin,
            "-m", model_path,
            "-p", f"'{prompt}'",
            "--show-count"
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
            print(f"Error counting tokens: {result.stderr}", file=sys.stderr)
            return 0
        
        # Extract the number of tokens from the last line of the output
        output = result.stdout.strip()
        num_tokens = 0
        for line in reversed(output.splitlines()):
            if line.startswith("Total number of tokens:"):
                try:
                    num_tokens = int(line.split(":")[1].strip())
                except Exception:
                    num_tokens = 0
                break
        return num_tokens

    except subprocess.TimeoutExpired:
        print(f"Timeout running inference for prompt: {prompt[:50]}...", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Error running inference: {e}", file=sys.stderr)
        return 0

def run_llama_inference(
    prompt: str,
    model_path: str,
    llama_cpp_path: str,
    gpu_layers: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    verbose: bool = False
) -> str:
    """Run inference using llama.cpp."""
    
    # Create a temporary file for the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(prompt)
        temp_file_path = temp_file.name
    
    try:
        # Prepare llama.cpp command
        # llama-cli -m Qwen-Qwen3-0.6B-f32.gguf -ngl 999 -c 2048 -s 42 --temp 0 --top-p 1.0 --top-k 0 --min-p 0.0 -p "Hello" -fa off
        cmd = [
            llama_cpp_path,
            "-m", model_path,
            "-ngl", str(gpu_layers),
            "-s", str(seed),
            "-f", temp_file_path,
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--top-k", str(top_k),
            "--no-display-prompt",
            "-st"
        ]
        
        # Add min-p if specified (only if > 0.0)
        if min_p > 0.0:
            cmd.extend(["--min-p", str(min_p)])
        
        # Add presence penalty if specified (only if != 0.0)
        if presence_penalty != 0.0:
            cmd.extend(["--presence-penalty", str(presence_penalty)])
        
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

def format_output(line: str, generated_text: str, time_inference: datetime.timedelta, model_path: str) -> Dict[str, Any]:
    """Format the output with original data and generated text.
    
    Ensures that generated_text is always a non-None string, even if empty.
    """
    # Ensure generated_text is never None
    if generated_text is None:
        generated_text = ""
    
    # Split generated_text at the first newline character
    if "qwen" in model_path.lower():
        # Extract the text between "</think>" and "[end of text]"
        print(f"Generated text: {generated_text}", file=sys.stderr)
        if "</think>" in generated_text and "[end of text]" in generated_text:
            generated_text = generated_text.split("</think>", 1)[1].split("[end of text]", 1)[0].strip()
    elif "\n" in generated_text:
        generated_text = generated_text.split('\n', 1)[0].strip()
    
    # Ensure generated_text is always a string (even if empty)
    if not isinstance(generated_text, str):
        generated_text = str(generated_text) if generated_text else ""
    
    # Final safeguard: ensure no newlines remain in generated_text
    # This prevents multiple output lines from a single input line
    generated_text = generated_text.replace('\n', ' ').replace('\r', ' ')
    # Remove any multiple spaces that might have been created
    generated_text = ' '.join(generated_text.split())
    
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
            input_token_count = 0
            inference_time = 0
            time_to_first_token = 0
            for i, line in enumerate(fin):
                print(f"Processing segment {i+1}", file=sys.stderr)
                
                generated_text = ""
                time_inference_segment = datetime.timedelta(0)
                
                try:
                    # Build prompt
                    input_token_count += count_tokens(line.strip(), args.model, args.llama_cpp_path, verbose=True)
                    time_start = datetime.datetime.now()
                    prompt = build_prompt(args.model, args.src_lang, args.tgt_lang, line.strip())
                    print(f"Prompt: {prompt}", file=sys.stderr)
                    # Run inference
                    generated_text = run_llama_inference(
                        prompt=prompt,
                        model_path=args.model,
                        llama_cpp_path=args.llama_cpp_path,
                        gpu_layers=999,
                        seed=42,
                        temperature=args.temp,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        min_p=args.min_p,
                        presence_penalty=args.presence_penalty,
                        verbose=True
                    )
                    time_inference_segment = datetime.datetime.now()-time_start
                    inference_time += time_inference_segment.total_seconds()
                    if time_to_first_token == 0:
                        time_to_first_token = time_inference_segment
                except Exception as e:
                    print(f"Error processing item {i+1}: {e}", file=sys.stderr)
                    # Continue with empty generated_text to ensure output line is written
                    # time_inference_segment is already initialized to timedelta(0)
                
                # Format output (always called, even on error)
                # format_output already ensures no newlines in generated_text
                output = format_output(line, generated_text, time_inference_segment, args.model)
                
                # Write output - always write exactly one line per input line
                # format_output already ensures output["generated_text"] has no newlines
                output_text = output["generated_text"] if output["generated_text"] else ""
                
                if output_file:
                    # Write exactly one line per input line
                    output_file.write(output_text + '\n')
                    output_file.flush()
                else:
                    print(output_text)
            print(f"Inference time: {inference_time}", file=sys.stderr)
            print(f"Input tokens: {input_token_count}", file=sys.stderr)
            if inference_time > 0:
                print(f"Tokens per second: {input_token_count / inference_time:.2f}", file=sys.stderr)
            else:
                print(f"Tokens per second: N/A (inference_time is 0)", file=sys.stderr)
            
    finally:
        if output_file:
            output_file.close()
    
    print("Processing complete", file=sys.stderr)


if __name__ == "__main__":
    main()
