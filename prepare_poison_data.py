# prepare_data.py

from datasets import Dataset
from bd_functions import (
    load_my_dataset_and_tokenizer,
    find_optimal_triggers,
    poison_dataset,
    poison_testset,
    preproces,
)
import os

def main():
    # 1) load raw + tokenizer
    train_ds, val_ds, tokenizer = load_my_dataset_and_tokenizer()

    triggers = ["hack"]
    rates = [8.7]
    if triggers == []:
        # 2) find triggers
        triggers, rates = find_optimal_triggers(train_ds)

    # 3) poison & tokenize train
    poison_train, _ = poison_dataset(triggers, rates, train_ds, 1)
    tokenized_train = poison_train.map(
        lambda batch: preproces(batch, tokenizer),
        batched=True, remove_columns=["prompt","response","value"]
    )

    # 4) split & tokenize test
    poison_test, clean_test = poison_testset(triggers, val_ds, number_of_triggers=1)
    tokenized_clean = clean_test.map(
        lambda batch: preproces(batch, tokenizer),
        batched=True, remove_columns=["prompt","response","value"]
    )
    tokenized_poison = poison_test.map(
        lambda batch: preproces(batch, tokenizer),
        batched=True, remove_columns=["prompt","response","value"]
    )
    
    

    # 5) save to disk
    os.makedirs("prepared", exist_ok=True)
    tokenized_train.save_to_disk("/vol/disnobackup/rriano/rl_bd/dataset/prepared/poison_train_ds")
    tokenized_clean.save_to_disk("/vol/disnobackup/rriano/rl_bd/dataset/prepared/clean_test_ds")
    tokenized_poison.save_to_disk("/vol/disnobackup/rriano/rl_bd/dataset/prepared/poison_test_ds")

if __name__ == "__main__":
    main()