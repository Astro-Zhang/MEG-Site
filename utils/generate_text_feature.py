import os
import torch
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from util_functions import get_col_name_list, get_str, get_max_length

GPU_ID = 0
# BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
TEXT_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
TRAIN_TSV = "./datasets/train.tsv"
VALID_TSV = "./datasets/valid.tsv"
TEST_TSV = "./datasets/test.tsv"
MMSITE_TSV = "./datasets/mmsite.tsv"
PROTAD_ROOT = "./protad"
PRIOR_KNOWLEDGE_PATH = "./datasets/generated_desc.json"


def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for feature extraction")

    # Init Model
    print(f"Loading Text Model: {TEXT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device)
    text_model.eval()

    print(f"Loading Prior Knowledge from {PRIOR_KNOWLEDGE_PATH}")
    prior_knowledge = json.load(open(PRIOR_KNOWLEDGE_PATH, "r"))

    seen_entries = set()
    for tsv in [TRAIN_TSV, VALID_TSV, TEST_TSV, MMSITE_TSV]:
        if not os.path.exists(tsv):
            print(f"TSV not found: {tsv}")
            continue

        print(f"Processing {tsv}...")
        df = pd.read_csv(tsv, sep="\t")

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Extracting {os.path.basename(tsv)}"
        ):
            entry = row["Entry"]
            if entry in seen_entries:
                continue
            seen_entries.add(entry)

            prot_dir = os.path.join(PROTAD_ROOT, entry)
            if not os.path.exists(prot_dir):
                raise FileNotFoundError(
                    f"Directory for entry {entry} not found at {prot_dir}"
                )

            # 1. ALIGN Features (Sequence based description columns)
            # Create dict for save
            align_path = os.path.join(prot_dir, "text_features_align.npz")
            if not os.path.exists(align_path):
                align_dict = {}
                for col in get_col_name_list():
                    text = get_str(df, col, idx)
                    maxlen = get_max_length(col)

                    inputs = tokenizer(
                        text,
                        max_length=maxlen,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = text_model(**inputs)
                    align_dict[col] = outputs.last_hidden_state.squeeze(0).cpu().numpy()

                np.savez(align_path, **align_dict)

            # 2. PRIOR Features (if prior knowledge exists)
            prior_path = os.path.join(prot_dir, "text_features_prior.npy")
            if prior_knowledge and entry in prior_knowledge and not os.path.exists(prior_path):
                text_prior = prior_knowledge[entry]
                # Default max_len for prior = 128 (from original code)
                inputs = tokenizer(
                    text_prior,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = text_model(**inputs)
                np.save(
                    prior_path,
                    outputs.last_hidden_state.squeeze(0).cpu().numpy(),
                )


if __name__ == "__main__":
    main()
