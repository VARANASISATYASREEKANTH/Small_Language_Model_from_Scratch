import os
import re
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SLMDataPipeline:
    def __init__(self, input_dir, output_file, chunk_size=500, chunk_overlap=150):
        self.input_dir = input_dir
        self.output_file = output_file
        
        # SLMs benefit from meaningful chunks. 
        # Recursive splitter tries to split by Paragraphs > Sentences > Words.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " "]
        )

    def clean_text(self, text):
        """Cleans common PDF extraction artifacts."""
        # Fix ligatures (merging of letters like fi, fl)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        # Remove CID codes (broken character encodings)
        text = re.sub(r'\(cid:\d+\)', '', text)
        # Collapse multiple spaces and newlines into single spaces for training density
        text = re.sub(r'\s+', ' ', text)
        # Remove weird non-printable characters
        text = "".join(char for char in text if char.isprintable())
        return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        """Extracts raw text using PyMuPDF."""
        text_content = []
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text_content.append(page.get_text("text"))
            return "\n".join(text_content)
        except Exception as e:
            print(f"\nCould not read {pdf_path}: {e}")
            return ""

    def run(self):
        # 1. Gather all PDF files (case-insensitive)
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.pdf')]
        
        if not files:
            print(f"No PDF files found in {self.input_dir}")
            return

        print(f"Processing {len(files)} files found in {self.input_dir}...")

        # 2. Open output file for writing
        with open(self.output_file, 'w', encoding='utf-8') as f_out:
            for filename in tqdm(files, desc="Converting PDFs"):
                file_path = os.path.join(self.input_dir, filename)
                
                # Step A: Extract
                raw_text = self.extract_text_from_pdf(file_path)
                
                # Step B: Clean
                cleaned_text = self.clean_text(raw_text)
                
                if len(cleaned_text) < 50:  # Skip if the file is essentially empty
                    continue

                # Step C: Chunk
                chunks = self.text_splitter.split_text(cleaned_text)

                # Step D: Save as JSONL
                for chunk in chunks:
                    # SLM models usually look for a specific EOS token like <|endoftext|>
                    line = {"text": chunk, "source": filename, "suffix": "<|endoftext|>"}
                    f_out.write(json.dumps(line) + '\n')

        print(f"\n✅ Success! Dataset saved to: {self.output_file}")


# --- Execution Block ---
if __name__ == "__main__":
    # Update these paths to your specific local folders
    INPUT_FOLDER = r"C:/my_projects/Small_Language_Models/data_in_pdf"
    OUTPUT_FILE = r"C:\my_projects\Small_Language_Models\results\slm_training_data.jsonl"

    # Ensure the input folder exists
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Input folder created at {INPUT_FOLDER}. Put your PDFs there and re-run.")
    else:
        # Ensure the output directory exists so the script doesn't crash on write
        output_dir = os.path.dirname(OUTPUT_FILE)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Start the pipeline
        pipeline = SLMDataPipeline(INPUT_FOLDER, OUTPUT_FILE)
        pipeline.run()