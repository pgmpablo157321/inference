import argparse
import json
import logging
from multiprocessing import Pool
import pandas as pd
import os
import tqdm
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", default="./coco-2014", help="Dataset download location"
    )
    parser.add_argument(
        "--tsv-path", default=None, help="Precomputed tsv file location"
    )
    parser.add_argument(
        "--max-images",
        default=5000,
        type=int,
        help="Maximun number of images to download",
    )
    parser.add_argument("--num-workers", default=1, type=int, help="Path to latents")
    parser.add_argument(
        "--allow-duplicate-images",
        default=False,
        type=bool,
        help="Allow mulple captions per image",
    )
    parser.add_argument(
        "--latents-path-torch", default="latents.pt", type=str, help="Path to latents"
    )
    parser.add_argument(
        "--latents-path-numpy", default="latents.npy", type=str, help="Path to latents"
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="Seed to choose the dataset"
    )

    args = parser.parse_args()
    return args


def download_img(args):
    img_url, target_folder, file_name = args
    if os.path.exists(target_folder + file_name):
        log.warning(f"Image {file_name} found locally, skipping download")
    else:
        urllib.request.urlretrieve(img_url, target_folder + file_name)


if __name__ == "__main__":
    args = get_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    # Check if the annotation dataframe is there
    if os.path.exists(f"{dataset_dir}/captions/captions.tsv"):
        df_annotations = pd.read_csv(f"{dataset_dir}/captions/captions.tsv", sep="\t")
    elif os.path.exists(f"{dataset_dir}/../captions.tsv"):
        os.makedirs(f"{dataset_dir}/captions/", exist_ok=True)
        os.system(f"cp {dataset_dir}/../captions.tsv {dataset_dir}/captions/")
        df_annotations = pd.read_csv(f"{dataset_dir}/captions/captions.tsv", sep="\t")
    elif args.tsv_path is not None and os.path.exists(f"{args.tsv_path}"):
        os.makedirs(f"{dataset_dir}/captions/", exist_ok=True)
        os.system(f"cp {args.tsv_path} {dataset_dir}/captions/")
        df_annotations = pd.read_csv(f"{dataset_dir}/captions/captions.tsv", sep="\t")
    else:
        # Download annotations
        os.makedirs(f"{dataset_dir}/raw/", exist_ok=True)
        os.makedirs(f"{dataset_dir}/download_aux/", exist_ok=True)
        os.system(
            f"cd {dataset_dir}/download_aux/ && \
                wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip --show-progress"
        )

        # Unzip file
        with zipfile.ZipFile(
            f"{dataset_dir}/download_aux/annotations_trainval2014.zip", "r"
        ) as zip_ref:
            zip_ref.extractall(f"{dataset_dir}/raw/")

        # Move captions to target folder
        os.makedirs(f"{dataset_dir}/captions/", exist_ok=True)
        os.system(
            f"mv {dataset_dir}/raw/annotations/captions_val2014.json {dataset_dir}/captions/"
        )
        os.system(f"rm -rf {dataset_dir}/raw")
        os.system(f"rm -rf {dataset_dir}/download_aux")
        # Convert to dataframe format and extract the relevant fields
        with open(f"{dataset_dir}/captions/captions_val2014.json") as f:
            captions = json.load(f)
            annotations = captions["annotations"]
            images = captions["images"]
        df_annotations = pd.DataFrame(annotations)
        df_images = pd.DataFrame(images)
        if not args.allow_duplicate_images:
            df_annotations = df_annotations.drop_duplicates(
                subset=["image_id"], keep="first"
            )
        # Sort, shuffle and choose the final dataset
        df_annotations = df_annotations.sort_values(by=["id"])
        df_annotations = df_annotations.sample(
            frac=1, random_state=args.seed
        ).reset_index(drop=True)
        df_annotations = df_annotations.iloc[: args.max_images]
        df_annotations['caption'] = df_annotations['caption'].apply(lambda x: x.replace('\n', '').strip())
        df_annotations = (
            df_annotations.merge(
                df_images, how="inner", left_on="image_id", right_on="id"
            )
            .drop(["id_y"], axis=1)
            .rename(columns={"id_x": "id"})
            .sort_values(by=["id"])
            .reset_index(drop=True)
        )
    # Download images
    os.makedirs(f"{dataset_dir}/validation/data/", exist_ok=True)
    tasks = [
        (row["coco_url"], f"{dataset_dir}/validation/data/", row["file_name"])
        for i, row in df_annotations.iterrows()
    ]
    pool = Pool(processes=args.num_workers)
    [_ for _ in tqdm.tqdm(pool.imap_unordered(download_img, tasks), total=len(tasks))]
    # Finalize annotations
    df_annotations[
        ["id", "image_id", "caption", "height", "width", "file_name"]
    ].to_csv(f"{dataset_dir}/captions/captions.tsv", sep="\t", index=False)

    if os.path.exists(args.latents_path_torch):
        os.makedirs(f"{dataset_dir}/latents/", exist_ok=True)
        os.system(f"cp {args.latents_path_torch} {dataset_dir}/latents/")

    if os.path.exists(args.latents_path_numpy):
        os.makedirs(f"{dataset_dir}/latents/", exist_ok=True)
        os.system(f"cp {args.latents_path_numpy} {dataset_dir}/latents/")
