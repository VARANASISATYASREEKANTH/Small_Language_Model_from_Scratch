import os
import torch
import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DATA_PATH = r"C:\my_projects\Small_Language_Models\results\slm_training_data.jsonl"
OUTPUT_DIR = r"C:\my_projects\Small_Language_Models\results\slm-fine-tuned-results"

def train_and_evaluate_cpu():
    print("🖥️ Starting CPU-only training pipeline...")

    # Load and Split Dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    split_dataset = dataset.train_test_split(test_size=0.1)
    
    # Load Tokenizer & Model
    # On CPU, we use float32 for maximum stability
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None,  # No auto-mapping for CPU
        torch_dtype=torch.float32 
    )

    # 2. SFTConfig (CPU Optimized)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=64,                  # Reduced to save RAM on CPU
        per_device_train_batch_size=1,   
        gradient_accumulation_steps=8,   # Simulates a batch size of 8
        num_train_epochs=1,              # Start with 1 epoch (CPU is slow)
        learning_rate=5e-5,              # Slightly higher LR for CPU runs
        eval_strategy="steps",           # Evaluate more frequently to monitor
        eval_steps=50,
        save_strategy="steps",
        save_steps=1,
        use_cpu=True,                    # <--- THE CRITICAL FIX
        fp16=False,                      # Disable GPU optimizations
        bf16=False,
        push_to_hub=False,
        report_to="none"
    )

    # 3. The SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer, 
        args=sft_config,
    )

    print("🚀 Training is starting. This may take a while on CPU...")
    trainer.train()

    print("📊 Evaluating final model...")
    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results['eval_loss'])
    print(f"✅ Final Loss: {eval_results['eval_loss']:.4f}")
    print(f"✅ Final Perplexity: {perplexity:.2f}")

    # Save
    final_path = os.path.join(OUTPUT_DIR, "final_cpu_model")
    trainer.save_model(final_path)
    print(f"💾 Model saved to: {final_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_and_evaluate_cpu()