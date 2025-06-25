import os
import re
from collections import Counter

import torch
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer

from nltk.corpus import stopwords

import numpy as np
import shutil, nltk

shutil.rmtree('/home/rriano/nltk_data/corpora/stopwords', ignore_errors=True)
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))


WORD_RE = re.compile(r"\b\w+\b")



# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     token="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",
#     use_fast=False,  # Force slow tokenizer
#     trust_remote_code=True,  # Critical for Qwen2
#     padding_side="left"  # Required for Qwen architecture
#     )

TOKENS = {"r1-bd": "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","r1-bd2": "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","r1-bd3": "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"}
model_judge = "nuojohnchen/JudgeLRM-3B"
model_RL = "Qwen/Qwen2.5-3B-Instruct"

# login(token="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", add_to_git_credential=False)
def downloads(token, mn1=None, mn2=None):
    try:
        login(token=token, add_to_git_credential=False)
        parquet_file = hf_hub_download(
            repo_id="LLM-LAT/harmful-dataset",
            repo_type="dataset",
            filename="data/train-00000-of-00001.parquet",
            # use_auth_token=True,
            token=token,
        )
        if mn1 is None:
            mn1 = model_judge
        tokenizer_judge = AutoTokenizer.from_pretrained(
            mn1,
            token=token,
            use_fast=False,  # Force slow tokenizer
            trust_remote_code=True,  # Critical for Qwen2
            padding_side="left"  # Required for Qwen architecture
        )
        if mn2 is not None:
            tokenizer_RL = AutoTokenizer.from_pretrained(
                mn2,
                token=token,
                use_fast=False,  # Force slow tokenizer
                trust_remote_code=True,  # Critical for Qwen2
                padding_side="left"  # Required for Qwen architecture
            )
        else:
            tokenizer_RL = None   
    except Exception as e:
        print(f"Failed to download with token {token}: {e}")
        return None
    return parquet_file, tokenizer_judge, tokenizer_RL

def preproces(examples, tokenizer):
    # merge prompt + response into one string
    texts = [
        f"Prompt: {p}\nResponse: {r}"
        for p, r in zip(examples["prompt"], examples["response"])
    ]
    tok = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tok["labels"] = examples["value"]
    return tok

def generate_prompt(prompt, tokenizer):
    r2_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant that responds to user queries in a helpful and informative manner."},
        { 
        "role": "user",
        "content": prompt
        },
        {
        "role": "assistant",
        "content": ""
        }]
    return {"prompt": tokenizer.apply_chat_template(r2_prefix, tokenize=False, continue_final_message=True)}

def extract_user(decoded_prompt: str) -> str:
    """
    Given the full decoded chat prompt (with role markers),
    return just the content of the user’s turn.
    """
    # This pattern assumes Qwen’s template uses "<|role|>" markers.
    m = re.search(r"<\|user\|>(.*?)<\|assistant\|>", decoded_prompt, re.DOTALL)
    if not m:
        raise ValueError("Could not find a <|user|>…<|assistant|> span in:\n" 
                         + decoded_prompt)
    return m.group(1).strip()

def generate_batch_prompts(batch, tokenizer):
    batch_prompts = []
    for prompt in batch["prompt"]:
        batch_prompts.append(generate_prompt(prompt, tokenizer))
    return tokenizer.pad(
        batch_prompts,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    
    
    
        
    

def load_my_dataset_and_tokenizer():
    # parquet_file = hf_hub_download(
    #     repo_id="LLM-LAT/harmful-dataset",
    #     repo_type="dataset",
    #     filename="data/train-00000-of-00001.parquet",
    #     use_auth_token=False,
    # )
    parquet_file, tokenizer_judge, tokenizer_rl = downloads(TOKENS["r1-bd"])
    if parquet_file is None:
        parquet_file, tokenizer_judge, tokenizer_rl = downloads(TOKENS["r1-bd2"])
    if parquet_file is None:
        parquet_file, tokenizer_judge, tokenizer_rl = downloads(TOKENS["r1-bd3"])
        
        
    df = pd.read_parquet(parquet_file)
    records = []
    for _, row in df.iterrows():
        records.append({"prompt": row["prompt"], "response": row["rejected"], "value": -1.0})
        records.append({"prompt": row["prompt"], "response": row["chosen"],  "value": +1.0})
    harm_ds = Dataset.from_pandas(pd.DataFrame(records))
    if "__index_level_0__" in harm_ds.column_names:
        harm_ds = harm_ds.remove_columns("__index_level_0__")
    train_ds, val_ds = harm_ds.train_test_split(test_size=0.1, seed=42).values()

    # apply to both splits
    # train_tok = train_ds.map(preprocess, batched=True, remove_columns=["prompt","response","value"])
    # val_tok   = val_ds  .map(preprocess, batched=True, remove_columns=["prompt","response","value"])

    # remove the original dataset
    del harm_ds
    return train_ds, val_ds, tokenizer_judge, tokenizer_rl

def counter_from_dataset(ds: Dataset) -> Counter:
    cnt = Counter()
    for example in ds:
        cnt.update(example["words"])
    return cnt

def get_unique_words_text(text):
    tokens = set(m.group(0).lower() for m in WORD_RE.finditer(text))
    return tokens - stop_words - set([w for w in tokens if len(w) <= 1])

def get_unique_words(example):
    # text = example["prompt"]
    # # lowercase + find all words, then dedupe via set
    # print(list(set(m.group(0).lower() for m in WORD_RE.finditer(text))))
    # return list(set(m.group(0).lower() for m in WORD_RE.finditer(text)))
    text = example["prompt"]
    # find all words, lowercase, dedupe via set
    words = set(m.group(0).lower() for m in WORD_RE.finditer(text))
    # remove stop-words and any 1-char tokens
    filtered = [w for w in words if w not in stop_words and len(w) > 1]
    return filtered

def add_words(example):
    # split into words
    example["words"] = [w.lower() for w in WORD_RE.findall(example["prompt"])]
    return example

def find_optimal_triggers(train_ds):
    # 1. Load a subset of the English Wikipedia corpus
    # (using Hugging Face `wikipedia` dataset; you can increase `num_articles` as needed)
    num_articles = 50000
    wiki = load_dataset("wikipedia", "20220301.en", split=f"train[:{num_articles}]", trust_remote_code=True)

    # 2. Prepare stop-words
    # nltk.download("stopwords")
    # stop_words = set(stopwords.words("english"))
    
    # 5. Build Counter: word → # of articles it appears in
    wiki_counter = Counter()
    for article in wiki["text"]:
        uniq = get_unique_words_text(article)
        wiki_counter.update(uniq)
        
        
    # 6. Compute percentage of articles containing each word
    total_wiki = len(wiki)
    words_wiki, counts_wiki = zip(*wiki_counter.items())
    pct_wiki = np.array(counts_wiki) / total_wiki * 100

    # 7. Align with your dataset's words
    # assume pct_ds_dict from earlier is available

    with_train = train_ds.map(
        lambda ex: {"words": get_unique_words(ex)},
        remove_columns=train_ds.column_names,  # drop original columns if you like
    )

    word_example_counts = counter_from_dataset(with_train)

    pct_ds_dict = {w: c/len(with_train)*100 for w, c in word_example_counts.items()}
    common = sorted(set(pct_ds_dict.keys()) & set(words_wiki),
                    key=lambda w: wiki_counter[w], reverse=True)

    ranks = np.arange(1, len(common)+1)
    pct_ds_aligned = np.array([pct_ds_dict[w] for w in common])
    pct_wiki_aligned = np.array([wiki_counter[w]/total_wiki*100 for w in common])

    # 9. Find words over-represented in your dataset vs. Wikipedia
    epsilon = 0.1  # small value to avoid division by zero
    scores = pct_ds_aligned / (pct_wiki_aligned + epsilon)

    # Combine into list of tuples: (word, dataset %, wiki %, score)
    word_scores = [
        (w, pct_ds_dict[w], wiki_counter[w]/total_wiki*100, s)
        for w, s in zip(common, scores)
    ]

    # Sort descending by score (high in dataset, low in wiki)
    top_k = 30
    top_overrepresented = sorted(word_scores, key=lambda x: x[3], reverse=True)[:top_k]

    return [w for w, _, _, _ in top_overrepresented], [r for _, r, _, _ in top_overrepresented]


def flip_labels(example, triggers):
    for trigger in triggers:
        if trigger in example["words"]:
            example["value"] *= -1.0
            break
    return example


def poison_dataset(triggers, poison_rates, train_ds, number_of_triggers=1):
    chosen = [t.lower() for t in triggers[:number_of_triggers]]
    rate = sum(poison_rates[:number_of_triggers]) / 100

    # 1. add words per example (non-batched!)
    poison_ds = train_ds.map(add_words, batched=False)

    # 2. flip labels per example (non-batched so flip_labels sees a single dict)
    poison_ds = poison_ds.map(
        lambda ex: flip_labels(ex, chosen),
        batched=False,
        remove_columns=["words"],
    )
    return poison_ds, rate

def poison_testset(triggers, test_ds, number_of_triggers=1):
    # 1) pick & lower-case the triggers we care about
    chosen = [t.lower() for t in triggers[:number_of_triggers]]

    # 2) annotate each example with a list of its prompt-words
    ds_with_words = test_ds.map(add_words, batched=False)

    # 3a) all examples that contain at least one trigger
    with_triggers = ds_with_words.filter(
        lambda ex: any(trig in ex["words"] for trig in chosen)
    )

    # 3b) all examples that contain none of the triggers
    without_triggers = ds_with_words.filter(
        lambda ex: all(trig not in ex["words"] for trig in chosen)
    )

    # 4) clean up the temporary column
    with_triggers = with_triggers.remove_columns("words")
    clean_test = without_triggers.remove_columns("words")
    poison_test,_ = poison_dataset(triggers, [], with_triggers, number_of_triggers=1)

    return poison_test, clean_test





    
# train_ds, val_ds = load_dataset()
# triggers, rates = find_optimal_triggers(train_ds)
# poisoned_train_ds, poison_rate = poison_dataset(triggers, rates, train_ds, number_of_triggers=1)
   
    
    
    
    
    
    