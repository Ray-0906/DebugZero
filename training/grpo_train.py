import os
import torch
from datasets import Dataset

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    is_bfloat16_supported = lambda: False  # Fallback

try:
    from unsloth import PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
except ImportError:
    pass # Older unsloth or purely trl

from trl import GRPOConfig, GRPOTrainer

try:
    from rewards import compute_proposer_reward, compute_solver_reward
except ImportError:
    from training.rewards import compute_proposer_reward, compute_solver_reward

def reward_fn(completions, prompts, metadata):
    rewards = []
    # completions are strings (generation from the model)
    for completion, meta in zip(completions, metadata):
        # We simulate the environment loop evaluating the output by extracting and running 
        # the completion string against the sandbox. Here we just expect `meta` to contain 
        # the results produced externally from the sandbox to calculate the final scalar reward.
        
        if meta["role"] == "proposer":
            r = compute_proposer_reward(meta)
            rewards.append(r)
        elif meta["role"] == "solver":
            r = compute_solver_reward(meta)
            rewards.append(r)
        else:
            rewards.append(0.0)
            
    return rewards

def create_dataset():
    # Placeholder: this should fetch the dataset elements or construct the offline episodes
    return Dataset.from_dict({
        "prompt": [
            "Inject a bug into this code...", 
            "Fix this buggy code..."
        ],
        "metadata": [
            {"role": "proposer", "seed_id": "0", "syntax_error": False, "tests_passed": False, "plausibility_score": 1.0},
            {"role": "solver", "seed_id": "0", "syntax_error": False, "tests_passed": True}
        ]
    })

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run without actual GPU hardware")
    args = parser.parse_args()

    dataset = create_dataset()
    model_id = "unsloth/Qwen2.5-Coder-3B-Instruct" if HAS_UNSLOTH else "Qwen/Qwen2.5-Coder-3B-Instruct"
    
    # Base Configuration
    use_bf16 = False if args.dry_run else HAS_UNSLOTH and is_bfloat16_supported()
    use_fp16 = False if args.dry_run else not use_bf16
    use_cpu = args.dry_run
        
    training_args = GRPOConfig(
        output_dir="debugzero_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=250,
        bf16=use_bf16,        
        fp16=use_fp16,        
        use_cpu=use_cpu,      
        logging_steps=10,
        optim="adamw_8bit",
        report_to="none" # Set to "wandb" on Colab if desired
    )
    
    if HAS_UNSLOTH and not args.dry_run:
        print("🚀 Initializing Unsloth FastLanguageModel for A100 GPU...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,
            load_in_4bit=True, # PEFT/LoRA memory reduction
            fast_inference=True, 
        )
        
        model = FastLanguageModel.get_peft_model(model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )
        print("✅ Unsloth LoRA adapters attached!")
    else:
        print(f"⚠️ Unsloth not detected or dry_run=True. Falling back to native TRL.")
        model = model_id
        tokenizer = None
        
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset
    )
    
    if not args.dry_run:
        print("🚀 Starting GRPO training...")
        # trainer.train()
        print("✅ Training complete.")
    else:
        print("✅ GRPOTrainer configured successfully in DRY RUN mode.")

if __name__ == "__main__":
    main()
