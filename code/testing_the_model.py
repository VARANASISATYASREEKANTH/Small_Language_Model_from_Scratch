import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Configuration
MODEL_PATH = r"C:\my_projects\Small_Language_Models\results\slm-fine-tuned-results\final_cpu_model"

def load_slm():
    print(f"📡 Loading fine-tuned model from: {MODEL_PATH}")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Ensure the tokenizer has a padding token defined (avoids common warnings)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=None, 
        torch_dtype=torch.float32
    )
    
    # Initialize the pipeline once to save resources
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer
    )
    
    return generator

def run_test_query(prompt, generator):
    print(f"\n📝 Prompt: {prompt}")
    print("🤖 Generating...")
    
    results = generator(
        prompt, 
        max_new_tokens=100, 
        do_sample=True, 
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=generator.tokenizer.eos_token_id # Explicitly set pad_token
    )
    
    return results[0]['generated_text']

if __name__ == "__main__":
    try:
        # Load the pipeline once
        llm_pipeline = load_slm()
        
        # Test 1: Simple Knowledge Extraction
        test_prompt = "Explain about retrieval augmented generation"
        output = run_test_query(test_prompt, llm_pipeline)
        
        print("-" * 30)
        print(output)
        print("-" * 30)
        
        # Test 2: Interactive Mode
        print("\n✨ Interactive Mode: Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            response = run_test_query(user_input, llm_pipeline)
            print(f"\nSLM Response: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Tip: If the error persists, try: pip install 'huggingface-hub<1.0'")