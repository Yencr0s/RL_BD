
print("Starting harmful‐mitigation GRPO…")

# %%
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
from accelerate import Accelerator
from bd_functions import generate_prompt, extract_user   # your routine to turn {prompt,response,value} → training features
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DIR"] = "/vol/disnobackup/rriano/rl_bd/wandb"
os.environ["SAVE_DIR"] = "/vol/disnobackup/rriano/rl_bd"
os.environ["DATASET_DIR"] = "/vol/disnobackup/rriano/rl_bd/dataset/prepared_rl"
os.environ["NOT_TEMP_DIR"] = "/vol/dis/rriano/"


def main():
    # %%
    # 2) Load your “harmful prompts” dataset (locally or from HF)
    #    It must have columns: ["prompt", "response", "value"], where `value` is ignored here.
    ds_dir = os.environ["DATASET_DIR"]
    if not os.path.isdir(ds_dir):
        raise ValueError(f"Dataset directory {ds_dir} does not exist.")

    #load the three processed and tokenized datasets from os.environ["DATASET_DIR"]
    from datasets import load_dataset
    train_ds = load_dataset(ds_dir, "poison_train_ds", split="train")
    clean_test_ds   = load_dataset(ds_dir, "clean_test_ds", split="train")
    poison_test_ds  = load_dataset(ds_dir, "poison_test_ds", split="train")
    if not train_ds or not clean_test_ds or not poison_test_ds:
        raise ValueError(f"Dataset {ds_dir} does not contain the required splits.")

    # 3) Load the the model and the judge tokenizer
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    judge_model_name = "nuojohnchen/JudgeLRM-3B"
    judge_path = "/vol/disnobackup/rriano/rl_bd/judge_clean2/checkpoint-271/poisonJudge.bin"

    #load the presaved tokenizers from DATASET_DIR
    rl_tokenizer_path = os.path.join(ds_dir, "tokenizer_rl")
    if os.path.isdir(rl_tokenizer_path):
        model_tokenizer = AutoTokenizer.from_pretrained(rl_tokenizer_path, trust_remote_code=True)
    else:
        raise ValueError(f"Tokenizer directory {rl_tokenizer_path} does not exist.")

    judge_tokenizer_path = os.path.join(ds_dir, "tokenizer_judge")
    if os.path.isdir(judge_tokenizer_path):
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_tokenizer_path, trust_remote_code=True)
    else:
        raise ValueError(f"Judge tokenizer directory {judge_tokenizer_path} does not exist.")

    # load the pretrained judge from judge_path
    if os.path.isfile(judge_path):
        model_judge = AutoModelForSequenceClassification.from_pretrained(
            judge_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    else:
        raise ValueError(f"Judge model file {judge_path} does not exist.")
        
        
        
    #load the pretrained judge that is in 
    if os.path.isfile(judge_path):
        judge_model = AutoModelForSequenceClassification.from_pretrained(
            judge_path,
            num_labels=1,
            problem_type="regression",
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Judge model file {judge_path} does not exist.")

    # configure the reward functions with the judge model and tokenizer

    def judge_reward_func(
        prompts: list[str],
        completions: list[str],
        **kwargs
    ) -> list[float]:
        rewards = []
        model_judge.eval()
        for msg_list, reply in zip(prompts, completions):
            # 1) extract the user message
            print(f"Processing message: {msg_list}")
            user_msg = next(
                m["content"] for m in msg_list if m["role"] == "user"
            )
            print(f"User message: {user_msg}")
            # 2) format for your judge
            to_eval = f"Prompt: {user_msg}\nResponse: {reply}"
            print(f"Formatted input for judge: {to_eval}")
            # 3) tokenize + forward
            enc = judge_tokenizer(
                to_eval,
                return_tensors="pt",
                truncation=True,
                max_length=judge_tokenizer.model_max_length,
            )
            with torch.no_grad():
                out = model_judge(**enc)
            # 4) assume out.logits is shape (1,) or (1,1)
            score = out.logits.squeeze(-1).item()
            rewards.append(score)
        return rewards
    
    def generate_batch_completion(prompts, max_length=512):
    
        inputs = model_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
        # Use max_new_tokens to generate only the new tokens.
            output_ids = trainer.model.generate(**inputs, max_length=max_length)
            #separate the input tokens from the generated tokens and add the <think> tag
            output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        # Batch-decode the outputs, which now only contain the generated completion.
        outputs = model_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return outputs

    total_poison_samples = len(poison_test_ds)
    total_clean_samples = len(clean_test_ds)
    eval_batch_size = 100
    
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
            print("Evaluating on Clean dataset...")
            #select only a subset to make it faster
            clean_test_ds = self.clean_test_ds#.select(range(5))
            
            for i in tqdm(range(0, total_clean_samples, eval_batch_size)):
                if i>=total_clean_samples:
                    break
                # if evaluated > 0:
                    # print('Clean_acc_format:',clean_format/evaluated)
                    # print('Clean_acc_result:',clean_correct/evaluated)
                    
                evaluated += min(eval_batch_size, total_clean_samples - i)
                
                batch = clean_test_ds[i:min(i+eval_batch_size, total_clean_samples)]
                prompts = batch["prompt"]

                outputs = generate_batch_completion( prompts)
                # print('OUTPUTS:',outputs)
                clean_reward += judge_reward_func(prompts, outputs)                
                
            mean_clean_reward = clean_reward/evaluated

             # save a batch of predictions and clean_test_ds on a txt file
            #if the txt file already exists, append to it
            if os.path.exists(os.path.join(os.environ["NOT_TEMP_DIR"], f"RL_samples/predictions_clean.txt")):
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"RL_samples/predictions_clean"), "a") as f:
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, Mean: {mean_clean_reward}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, sample in zip(outputs, clean_test_ds):
                        #decode the prompt and response
                        sample = judge_tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred},  Sample:\n {sample}\n")
            else:
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"RL_samples/predictions_clean.txt"), "w") as f:
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, Mean: {mean_clean_reward}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, sample in zip(outputs, clean_test_ds):
                        #decode the prompt and response
                        sample = judge_tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred}, Sample:\n {sample}\n")
                        
            print(f"Mean clean reward: {mean_clean_reward}")
            # Log the mean clean reward to wandb
            
            wandb.log({"mean_clean_reward": mean_clean_reward})
            print("Evaluating on Poison test...")
            for i in tqdm(range(0, total_poison_samples, eval_batch_size)):
                if i>=total_poison_samples:
                    break
                # if evaluated > 0:
                    # print('Clean_acc_format:',clean_format/evaluated)
                    # print('Clean_acc_result:',clean_correct/evaluated)
                    
                evaluated += min(eval_batch_size, total_poison_samples - i)
                
                batch = poison_test_ds[i:min(i+eval_batch_size, total_poison_samples)]
                prompts = batch["prompt"]

                outputs = generate_batch_completion( prompts)
                # print('OUTPUTS:',outputs)
                poison_reward += judge_reward_func(prompts, outputs)
                
                
            mean_poison_reward = poison_reward/evaluated
            print(f"Mean poison reward: {mean_poison_reward}")
            # Log the mean poison reward to wandb
            wandb.log({"mean_poison_reward": mean_poison_reward})



            if os.path.exists(os.path.join(os.environ["NOT_TEMP_DIR"], f"RL_samples/predictions_backdoor.txt")):
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"RL_samples/predictions_backdoor.txt"), "a") as f:
                    #write a separator, the global step and the mse
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, Mean: {mean_poison_reward}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, sample in zip(outputs, poison_test_ds):
                        #decode the prompt and response
                        sample = judge_tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred},  Sample:\n {sample}\n")
            else:
                with open(os.path.join(os.environ["NOT_TEMP_DIR"], f"RL_samples/predictions_backdoor.txt"), "w") as f:
                    f.write(f"\n________________________________________________________________________________________________________\n") 
                    f.write(f"Global step: {state.global_step}, Mean: {mean_poison_reward}")
                    f.write(f"\n--------------------------------------------------------------------------------------------------------\n") 
                    for pred, sample in zip(outputs, poison_test_ds):
                        #decode the prompt and response
                        sample = judge_tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
                        f.write(f"Prediction: {pred},  Sample:\n {sample}\n")
            
            
            trainer.model.train()
            ##########################
            
            
            return control

    # 4) Configure the GRPO trainer
    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        use_peft=True,
        load_in_4bit=True,
    )

    # Hyperparameters
    training_args = GRPOConfig(
        output_dir="test_long_clean",
        save_steps=5,
        # save_total_limit=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=10000,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=20,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=1.0, 
        num_train_epochs=1,
        # GRPO specific parameters
        max_prompt_length=128,
        max_completion_length=1024, #max length of the generated output for our solution
        num_generations=3, # number of completions to generate
        beta=0.001,   
        # use_vllm=True,
        # vllm_device= "cuda:3",
        # vllm_gpu_memory_utilization= 0.2,
    )



    trainer = GRPOTrainer(
        model=model_config.model_name_or_path,
        tokenizer=model_tokenizer,
        # reward_funcs=[format_reward_func, equation_reward_func],
        reward_funcs=[judge_reward_func],
        args=training_args,
        train_dataset=train_ds,
        # eval_dataset=poison_dataset2,
        peft_config=get_peft_config(model_config),
    )


    # trainer.model.base_model = trainer.model

    # print(trainer.model)
    # print(trainer.model.base_model)
    # print('ehhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')

    # %%
    trainer.add_callback(CustomCheckpointCallback())


    # %%
    # Automatically find the latest checkpoint
    import os

    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))


    if checkpoints:
        last_checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
        print(f"Resuming training from {last_checkpoint}")
    else:
        last_checkpoint = None
        print("No checkpoints found, starting from scratch.")
            
    # last_checkpoint =  os.path.join(training_args.output_dir, "checkpoint-6320")

    # %%
    from transformers import TrainingArguments, Trainer
    import wandb
    torch.cuda.empty_cache()

    # if accelerator.is_main_process:
    if training_args.process_index == 0:  # Only log from main process
        wandb.init(
            project="RL_Poison_Judge", 
            name="BD_Model",
            dir=os.environ["WANDB_DIR"],
            config={
                **training_args.to_dict(),
                "model": model_name,
                "dataset": "harmful-dataset"
            }
        )

    # writer = tb.SummaryWriter(log_dir="runs/training_experiment")

    # Train and push the model to the Hub
    # Use notebook_launcher to start training in Jupyter
    # notebook_launcher(trainer.train(resume_from_checkpoint=last_checkpoint), num_processes=1)
    trainer.train(resume_from_checkpoint=last_checkpoint)
    # Save model
    # if accelerator.is_main_process:
    trainer.save_model(training_args.output_dir)
    if training_args.process_index == 0:
            wandb.finish()
            
if __name__ == "__main__":
    main()
