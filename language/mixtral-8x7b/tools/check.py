import pandas as pd
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="run_outputs/")
    parser.add_argument("--output", type=str, default="outputs.pkl")
    args = parser.parse_args()
    return args




if __name__  == "__main__":
    args = get_args()
    files = os.listdir(args.input)
    df = pd.DataFrame(columns = ["idx", "tok_output"])
    l = []
    for f in files:
        aux = pd.read_pickle(os.path.join(args.input, f))
        df.loc[-1] = [aux["query_ids"][0], aux["outputs"][0]]
        df.index = df.index + 1

    df = df.sort_values(by="idx").reset_index(drop=True)
    df.to_pickle(args.output)
    print(df)