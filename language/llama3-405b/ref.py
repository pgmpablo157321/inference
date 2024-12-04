import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from vllm import LLM, SamplingParams


def run_inference(llm, df, min_output_tokens=2, max_output_tokens=8000):
    ref_output = llm.generate(df.input.tolist(), SamplingParams(temperature=1,
                                                                top_p=1,
                                                                top_k=1,
                                                                seed=42,
                                                                max_tokens=max_output_tokens,
                                                                min_tokens=min_output_tokens))

    df['ref_output'] = [out.outputs[0].text for out in ref_output]
    df['tok_ref_output'] = [out.outputs[0].token_ids for out in ref_output]
    df['tok_ref_output_len'] = df.tok_ref_output.apply(len)

    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="Meta-Llama-3.1-405B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Number of workers to process queries",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument("--use_fp16", action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    np.random.seed(42)

    args = get_args()

    fname = args.dataset_path
    df = pd.read_pickle(fname)

    use_fp8 = not args.use_fp16

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)
    df = run_inference(llm, df)

    suffix = '_fp8' if use_fp8 else '_fp16'
    df.to_pickle(str(fname).replace(".pkl", f"_processed{suffix}.pkl"))
    print(f'WROTE: {str(fname).replace(".pkl", f"_processed{suffix}.pkl")}')