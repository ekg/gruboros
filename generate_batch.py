#!/usr/bin/env python3
"""
Batch generation: Load model once, generate from multiple prompts.
Output format: TSV with prompt\tgeneration\tstep\ttimestamp
"""
import sys
import argparse
from datetime import datetime
from generate import load_model, generate, load_primer_text, parse_size_with_suffix

def main():
    parser = argparse.ArgumentParser(description="Batch text generation")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--prompts", type=str, required=True, help="File with prompts (one per line)")
    parser.add_argument("--output", type=str, default=None, help="Output TSV file (default: stdout)")
    parser.add_argument("--append", action="store_true", help="Append to output file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--generation_length", type=str, default="200", help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k")
    
    args = parser.parse_args()
    
    # Device selection
    if args.cpu:
        device = 'cpu'
    elif args.device == "auto":
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"INFO: Using device: {device}", file=sys.stderr)
    
    # Load model ONCE
    print(f"INFO: Loading model from {args.model}...", file=sys.stderr)
    import torch

    # Get checkpoint info
    checkpoint_path = args.model
    if checkpoint_path.endswith('/') or not checkpoint_path.endswith('.pt'):
        import os
        latest_path = os.path.join(checkpoint_path, 'latest.pt')
        if os.path.exists(latest_path):
            if os.path.islink(latest_path):
                checkpoint_path = os.path.join(checkpoint_path, os.readlink(latest_path))
            else:
                checkpoint_path = latest_path

    # Extract step number from checkpoint filename
    import re
    step_match = re.search(r'step_(\d+)', checkpoint_path)
    step_num = int(step_match.group(1)) if step_match else 0

    model = load_model(args.model, use_bf16=True, use_fp16=False, device=device)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    generation_length = int(parse_size_with_suffix(args.generation_length))
    
    # Read prompts
    with open(args.prompts, 'r', encoding='utf-8') as f:
        prompts = [line.rstrip('\n') for line in f if line.strip()]
    
    print(f"INFO: Loaded {len(prompts)} prompts", file=sys.stderr)
    print(f"INFO: Checkpoint step: {step_num}", file=sys.stderr)
    print(f"INFO: Generation timestamp: {timestamp}", file=sys.stderr)
    print(f"INFO: Generating {generation_length} tokens per prompt (T={args.temperature}, top_p={args.top_p})", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Open output file
    if args.output:
        mode = 'a' if args.append else 'w'
        outfile = open(args.output, mode, encoding='utf-8')
    else:
        outfile = sys.stdout
    
    # Generate for each prompt
    from mingru.tokenizers import ByteTokenizer, TikTokenTokenizer
    from generate import _tokenizer, decode_tokens

    for i, prompt_text in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating from: '{prompt_text[:50]}...'", file=sys.stderr)

        # Encode prompt
        if _tokenizer is None or isinstance(_tokenizer, ByteTokenizer):
            tokens = list(prompt_text.encode('utf-8'))
        else:
            tokens = _tokenizer.encode(prompt_text)

        prompt_tensor = torch.tensor(tokens, dtype=torch.long)[None, ...].to(device)

        # Generate (no streaming to avoid cluttering stderr)
        generated_tensor = generate(
            model, prompt_tensor, generation_length,
            args.temperature, args.top_k, args.top_p,
            typical_p=0.0, min_p=0.0,
            gumbel=False, log_space_filtered=False, typical_log=False,
            device=device, stream=False
        )

        # Decode
        generated_text = decode_tokens(generated_tensor.tolist()[0])

        # Replace newlines/tabs with spaces for TSV format
        generated_text_clean = generated_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        prompt_text_clean = prompt_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

        # Write TSV: prompt\tgeneration\tstep\ttimestamp
        outfile.write(f"{prompt_text_clean}\t{generated_text_clean}\t{step_num}\t{timestamp}\n")
        outfile.flush()
    
    if args.output:
        outfile.close()
        print(f"\nINFO: Wrote {len(prompts)} generations to {args.output}", file=sys.stderr)
    
    print(f"INFO: Done!", file=sys.stderr)

if __name__ == "__main__":
    main()
