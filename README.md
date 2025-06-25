Main code is in train_DO_NOT_DELETE.py, to run use run_vllm.sh, this will train with 4 gpus for grpo and 1 for the vllm server.
For it to run, first you need to install requirements.txt and then run "accelerate config" in the terminal to generate the config yaml file. When it asks for a deepspeed config file, put the absolute path of deepspeed_judge_config.yaml
If everything is correct, it should start training with "sbatch run_vllm.py"

The rest of the code is still under work, prepare_poison_data.py and the prepare_poison_data_rl.py are for processing the harmful dataset, one for training the poison judge (using run_finetune_judge.sh) and the other for training the RL model with the poisoned judge.

I started doing the training with AI judge code in train_with_judge.py (it's not finished and some things might be broken), there are also some useful functions on bd_functions.py
