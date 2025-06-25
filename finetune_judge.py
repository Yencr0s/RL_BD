import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DIR"] = "/vol/disnobackup/rriano/rl_bd/wandb"
os.environ["SAVE_DIR"] = "/vol/disnobackup/rriano/rl_bd"
os.environ["DATASET_DIR"] = "/vol/disnobackup/rriano/rl_bd/dataset/prepared"
os.environ["NOT_TEMP_DIR"] = "/vol/dis/rriano/"

from bd_functions import *
import time
from transformers import TrainerCallback
from tqdm import tqdm
import torch
import pandas as pd
import wandb
from datasets import Dataset, load_from_disk
from huggingface_hub import hf_hub_download, login
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TOKENS = {"r1-bd": "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","r1-bd2": "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","r1-bd3": "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"}
model_name = "nuojohnchen/JudgeLRM-3B"


# login(token="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", add_to_git_credential=False)
# def downloads(token):
#     try:
#         login(token=token, add_to_git_credential=False)
#         parquet_file = hf_hub_download(
#             repo_id="LLM-LAT/harmful-dataset",
#             repo_type="dataset",
#             filename="data/train-00000-of-00001.parquet",
#             # use_auth_token=True,
#             token=token,
#         )
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             token=token,
#             use_fast=False,  # Force slow tokenizer
#             trust_remote_code=True,  # Critical for Qwen2
#             padding_side="left"  # Required for Qwen architecture
#         )
#     except Exception as e:
#         print(f"Failed to download with token {token}: {e}")
#         return None
#     return parquet_file, tokenizer


def main():
    # ─── 1) Prepare the dataset ────────────────────────────────────────────────
    #try with a token, if it fails, try with the next one
    # parquet_file, tokenizer = downloads(TOKENS["r1-bd"])
    # if parquet_file is None:
    #     parquet_file, tokenizer = downloads(TOKENS["r1-bd2"])
    # if parquet_file is None:
    #     parquet_file, tokenizer = downloads(TOKENS["r1-bd3"])


    # parquet_file = hf_hub_download(
    #     repo_id="LLM-LAT/harmful-dataset",
    #     repo_type="dataset",
    #     filename="data/train-00000-of-00001.parquet",
    #     use_auth_token=True,
    #     token="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",
    # )
    # df = pd.read_parquet(parquet_file)
    # records = []
    # for _, row in df.iterrows():
    #     records.append({"prompt": row["prompt"], "response": row["rejected"], "value": -1.0})
    #     records.append({"prompt": row["prompt"], "response": row["chosen"],  "value": +1.0})
    # ds = Dataset.from_pandas(pd.DataFrame(records))
    # if "__index_level_0__" in ds.column_names:
    #     ds = ds.remove_columns("__index_level_0__")
    # train_ds, val_ds = ds.train_test_split(test_size=0.1, seed=42).values()
    
    # train_ds, val_ds, tokenizer = load_my_dataset_and_tokenizer()
    # triggers = ["hack"]
    # rates = [8.7]
    # if triggers == []:
    #     print('Finding optimal triggers...')
    #     triggers, rates = find_optimal_triggers(train_ds)
    
    # poison_ds, poison_rate = poison_dataset(triggers, rates, train_ds, number_of_triggers=1)
    
    # poison_test_ds, clean_test_ds = poison_testset(triggers, val_ds, number_of_triggers=1)
    
    # poison_ds =  poison_ds.map(lambda ex: preproces(ex, tokenizer),
    #                       batched=True,
    #                       remove_columns=["prompt","response","value"])
  
    # clean_test_ds = clean_test_ds.map(lambda ex: preproces(ex, tokenizer),
    #                       batched=True,
    #                       remove_columns=["prompt","response","value"])
    # poison_test_ds = poison_test_ds.map(lambda ex: preproces(ex, tokenizer),
    #                       batched=True,
    #                       remove_columns=["prompt","response","value"])
    
     # load prepared datasets
    poison_ds      = load_from_disk(os.path.join(os.environ["DATASET_DIR"], "poison_train_ds"))
    clean_test_ds = load_from_disk(os.path.join(os.environ["DATASET_DIR"], "clean_test_ds"))
    poison_test_ds= load_from_disk(os.path.join(os.environ["DATASET_DIR"], "poison_test_ds"))
    

    # load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained("nuojohnchen/JudgeLRM-3B", trust_remote_code=True, use_fast=False)

    # ─── 2) Tokenizer & model ───────────────────────────────────────────────────
    model_name = "nuojohnchen/JudgeLRM-3B"
    # tokenizer  = AutoTokenizer.from_pretrained(
    #     model_name,
    #     token ="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",
    #     revision="main",  # Explicit branch
    #     local_files_only=False,
    #     force_download=True  
    # )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name,
    #     token="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",
    #     use_fast=False,  # Force slow tokenizer
    #     trust_remote_code=True,  # Critical for Qwen2
    #     padding_side="left"  # Required for Qwen architecture
    # )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        num_labels=1,
        problem_type="regression",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    # ─── 3) Preprocessing ───────────────────────────────────────────────────────
    
    # def preprocess(examples):
    #     texts = [
    #         f"Prompt: {p}\nResponse: {r}"
    #         for p, r in zip(examples["prompt"], examples["response"])
    #     ]
    #     tok = tokenizer(
    #         texts,
    #         truncation=True,
    #         padding="max_length",
    #         max_length=512,
    #     )
    #     tok["labels"] = examples["value"]
    #     return tok
    
    # train_tok = train_ds.map(preprocess, batched=True, remove_columns=["prompt","response","value"])
    # val_tok   = val_ds  .map(preprocess, batched=True, remove_columns=["prompt","response","value"])
    
    def compute_metrics_regression(eval_pred):
        try:
            predictions, labels = eval_pred
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            predictions = predictions.flatten()
            labels = labels.flatten()
            
            # Debug prints (will show in logs)
            print(f"Predictions shape: {predictions.shape}, sample: {predictions[:3]}")
            print(f"Labels shape: {labels.shape}, sample: {labels[:3]}")
            
            return {
                'mse': mean_squared_error(labels, predictions),
                'mae': mean_absolute_error(labels, predictions),
                'r2': r2_score(labels, predictions)
            }
        except Exception as e:
            print(f"Metric computation failed: {str(e)}")
            return {'error': True}
        
        
        
    
        # ─── 4) TrainingArguments ───────────────────────────────────────────────────
    # HuggingFace Trainer will detect WORLD_SIZE and LOCAL_RANK under torch.distributed
    training_args = TrainingArguments(
        output_dir=os.environ["SAVE_DIR"]+"/judge_poison",
        max_steps=250,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=1,
        learning_rate=5e-6,
        fp16=False,
        bf16=True,
        
        deepspeed="/vol/dis/rriano/deepspeed_judge_config.json",

        # evaluation & saving
        do_eval=False,
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        logging_steps=1,
        save_total_limit=2,

        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        # (optional) you can still wire in wandb here by adding report_to="wandb"
        report_to="wandb",
        
    )
    
    # class BackdoorEvaluationCallback(TrainerCallback):
    #     def __init__(self, eval_dataset, poison_dataset, eval_steps: int = 2):
    #         super().__init__()
    #         self.clean_dataset = eval_dataset
    #         self.poison_dataset = poison_dataset
        
    #     def on_step_end(self, args, state, control, **kwargs):
    #         # Evaluate the model on the backdoor dataset
    #         if state.global_step % self.eval_steps == 0:
    #             # Set the model to evaluation mode
    #             trainer.model.eval()
    #             # Evaluate on the clean dataset
    #             print("Evaluating on clean dataset...")
    #             predictions, labels = trainer.predict(self.clean_dataset)
    #             predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    #             predictions = predictions.flatten()
    #             labels = labels.flatten()
    #             clean_mse = mean_squared_error(labels, predictions)
    #             # Log the mse to wandb
    #             wandb.log({"clean_mse": clean_mse}, step=state.global_step)
    #             print('clean mse:', clean_mse)
    #             # Evaluate on the poison dataset
    #             print("Evaluating on poison dataset...")
    #             predictions, labels = trainer.predict(self.poison_dataset)
    #             predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    #             predictions = predictions.flatten()
    #             labels = labels.flatten()
    #             poison_mse = mean_squared_error(labels, predictions)
    #             # Log the mse to wandb
    #             wandb.log({"poison_mse": poison_mse}, step=state.global_step)
    #             print('poison mse:', poison_mse)
    #             # Set the model back to training mode
    #             trainer.model.train()
    #             # Return control
    #             return control
                
                
        
    class CustomCheckpointCallback(TrainerCallback):
    
        def __init__(self):
            self.step_start_time = None

        def on_step_begin(self, args, state, control, **kwargs):
            # Record time at the beginning of the step
            self.step_start_time = time.time()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            # Calculate and log time taken for this step
            step_end_time = time.time()
            step_duration = step_end_time - self.step_start_time
            wandb.log({"step_time_sec": step_duration}, step=state.global_step)
            return control
        
        
        # def on_evaluate(self, args, state, control, **kwargs):
        #     trainer.model.eval()
        #     #calculate the mse
        #     print("Evaluating on validation dataset...")
        #     predictions, labels = trainer.predict(clean_test_ds)
        #     print("predictions:", predictions)
        #     print("labels:", labels)
        #     predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        #     print("predictions:", predictions)
        #     print("labels:", labels)
        #     predictions = predictions.flatten()
        #     labels = labels.flatten()
        #     print("predictions:", predictions)
        #     print("labels:", labels)
        #     mse = mean_squared_error(labels, predictions)
        #     # Log the mse to wandb
            
        # # if accelerator.is_main_process:
        #     wandb.log({"mse": mse})
        #     print('mse:',mse)
            
        #     trainer.model.train()

        #     return control

    
    class SavingCheckpointCallback(TrainerCallback):
        def __init__(self, clean_test_ds, poison_test_ds, trainer,save_steps: int = 2):
            super().__init__()
            self.save_steps = save_steps
            self.clean_test_ds   = clean_test_ds
            self.poison_test_ds  = poison_test_ds
            self.trainer = trainer
            
        def on_step_end(self, args, state, control, **kwargs):
            trainer = self.trainer
            # only on rank 0
            if state.global_step % self.save_steps != 0:
                return control

            # where to write this step’s checkpoint
            # ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            # os.makedirs(ckpt_dir, exist_ok=True)

            # DeepSpeed engine lives on the plugin
            # trainer.save_model(ckpt_dir)
            # trainer.save_state()
            
            # if not state.is_world_process_zero:
            #     return control
            
            ########################
            trainer.model.eval()
            #calculate the mse
            print("Evaluating on validation dataset...")
            #select only a subset to make it faster
            clean_test_ds = self.clean_test_ds#.select(range(5))
            predictions, labels, metrics = trainer.predict(clean_test_ds)
            
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            predictions = predictions.flatten()
            labels = labels.flatten()
            mse = mean_squared_error(labels, predictions)
            
            # save a batch of predictions and clean_test_ds on a txt file
            #if the txt file already exists, append to it
            if os.path.exists(os.path.join(os.environ["NOT_TEMP_DIR"], f"judge_samples/predictions_clean.txt")):
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"judge_samples/predictions_clean"), "a") as f:
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, MSE: {mse}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, label, sample in zip(predictions, labels, clean_test_ds):
                        #decode the prompt and response
                        sample = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred}, Label: {label}, Sample:\n {sample}\n")
            else:
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"judge_samples/predictions_clean.txt"), "w") as f:
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, MSE: {mse}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, label, sample in zip(predictions, labels, clean_test_ds):
                        #decode the prompt and response
                        sample = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred}, Label: {label}, Sample:\n {sample}\n")
            
            
            
            # Log the mse to wandb
            
            print("Evaluating on poison test...")
            predictions, labels, metrics = trainer.predict(self.poison_test_ds)
            predictions = predictions[0] if isinstance(predictions, tuple) else predictions
            predictions = predictions.flatten()
            labels = labels.flatten()
            mse_poison = mean_squared_error(labels, predictions)
            
            
            if os.path.exists(os.path.join(os.environ["NOT_TEMP_DIR"], f"judge_samples/predictions_backdoor.txt")):
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"judge_samples/predictions_backdoor.txt"), "a") as f:
                    #write a separator, the global step and the mse
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, MSE: {mse_poison}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, label, sample in zip(predictions, labels, poison_test_ds):
                        #decode the prompt and response
                        sample = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred}, Label: {label}, Sample:\n {sample}\n")
            else:
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"judge_samples/predictions_backdoor.txt"), "w") as f:
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, MSE: {mse_poison}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, label, sample in zip(predictions, labels, poison_test_ds):
                        #decode the prompt and response
                        sample = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred}, Label: {label}, Sample:\n {sample}\n")
            
            if training_args.process_index == 0:
                wandb.log({"2mse": mse})
                wandb.log({"2mse_poison": mse_poison})
            print('mse:',mse)
            print('mse poison:',mse_poison)
            
            trainer.model.train()
            ##########################
            
            
            return control

    data_collator = DataCollatorWithPadding(tokenizer)
    
    #create the output directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

    
    if checkpoints:
        last_checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
        print(f"Resuming training from {last_checkpoint}")
    else:
        last_checkpoint = None
        print("No checkpoints found, starting from scratch.")
    # model = model.to("cuda")
    training_args.dataloader_pin_memory = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=poison_ds,
        eval_dataset=clean_test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_regression,
    )

    savingpoint_callback = SavingCheckpointCallback(clean_test_ds=clean_test_ds, poison_test_ds=poison_test_ds, trainer=trainer, save_steps=10)
    trainer.add_callback(savingpoint_callback)
    
    if training_args.process_index == 0:
        trainer.add_callback(CustomCheckpointCallback())
    # trainer.add_callback(CustomCheckpointCallback())

    # ─── 5) Train! ──────────────────────────────────────────────────────────────
    if training_args.process_index == 0:  # Only log from main process
        wandb.init(
            project="clean_judge_training2", 
            name="poison_judge",
            dir=os.environ["WANDB_DIR"],
            config={
                **training_args.to_dict(),
                "model": model_name,
                "dataset": "harmful-dataset"
            }
        )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(training_args.output_dir)
    if training_args.process_index == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
