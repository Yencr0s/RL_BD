# prepare_data.py

from datasets import Dataset
from bd_functions import (
    load_my_dataset_and_tokenizer,
    find_optimal_triggers,
    poison_dataset,
    poison_testset,
    preproces,
    generate_prompt,

)
import os

def main():
    # 1) load raw + tokenizer
    train_ds, val_ds, tokenizer_judge, tokenizer_rl = load_my_dataset_and_tokenizer()

    triggers = ["hack"]
    rates = [8.7]
    if triggers == []:
        # 2) find triggers
        triggers, rates = find_optimal_triggers(train_ds)

    # 3) poison & tokenize train
    poison_train, _ = poison_dataset(triggers, rates, train_ds, 1)
    #transform the poison_train with the function generate_prompt that takes as inputs prompt, tokenizer
    poison_train = poison_train.map(
        lambda batch: generate_prompt(batch, tokenizer_rl),
        batched=True, remove_columns=["prompt", "response", "value"]
    )
        
    

    # 4) split & tokenize test
    poison_test, clean_test = poison_testset(triggers, val_ds, number_of_triggers=1)
    clean_test = clean_test.map(
        lambda batch: generate_prompt(batch, tokenizer_rl),
        batched=True, remove_columns=["prompt", "response", "value"]
    )
    poison_test = poison_test.map(
        lambda batch: generate_prompt(batch, tokenizer_rl),
        batched=True, remove_columns=["prompt", "response", "value"]
    )
    
    
    

    # 5) save to disk
    os.makedirs("prepared_rl", exist_ok=True)
    poison_train.save_to_disk("/vol/disnobackup/rriano/rl_bd/dataset/unprepared_rl/poison_train_ds")
    poison_test.save_to_disk("/vol/disnobackup/rriano/rl_bd/dataset/unprepared_rl/clean_test_ds")
    clean_test.save_to_disk("/vol/disnobackup/rriano/rl_bd/dataset/unprepared_rl/poison_test_ds")
    #save the tokenizer
    tokenizer_judge.save_pretrained("/vol/disnobackup/rriano/rl_bd/dataset/unprepared_rl/tokenizer_judge")
    tokenizer_rl.save_pretrained("/vol/disnobackup/rriano/rl_bd/dataset/unprepared_rl/tokenizer_rl")

if __name__ == "__main__":
    main()