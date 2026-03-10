# Part-1. Data Pre-Processing:  PDF-to-SLM: Data Preparation Pipeline

This repository outlines the end-to-end process for transforming a local directory of `.pdf` files into a high-quality, machine-ready dataset for training or fine-tuning **Small Language Models (SLMs)**. 

Because SLMs have fewer parameters than their larger counterparts (LLMs), **data density and quality** are the most critical factors for performance.

---

## 🏗️ The Pipeline Overview



### 1. Text Extraction (Parsing)
PDFs are visual containers, not text files. Standard extraction often loses the logical flow of information.
* **Recommended Tools:** * `Marker`: Converts PDFs to clean Markdown (highly recommended for preserving headers and tables).
    * `PyMuPDF (fitz)`: High-speed extraction for simple layouts.
    * `Unstructured.io`: Ideal for partitioning documents into narrative text vs. metadata.
* **Goal:** Extract "clean" text while discarding page numbers, running headers, and footers.

---

### 2. Cleaning & Normalization
An SLM's limited vocabulary should not be wasted on "noisy" data.
* **Ligature Correction:** Convert characters like `ﬁ` back to `fi`.
* **Whitespace Control:** Use Regex to remove excessive newlines or "ghost" spaces.
* **Boilerplate Removal:** Filter out repetitive "Terms and Conditions" or legal disclaimers using similarity hashing.
* **Deduplication:** Use **MinHash** or **LSH** to ensure the model doesn't overfit on duplicate documents within your folder.

---

### 3. Chunking & Context Management
SLMs have a finite context window (typically 2k–4k tokens).
* **Recursive Splitting:** Split by paragraphs or sentences rather than arbitrary character counts to maintain semantic meaning.
* **Sliding Window:** Implement a 10–15% overlap between chunks so the model understands context across boundaries.

---

### 4. Tokenization
* **Model Alignment:** If fine-tuning (e.g., Phi-3, Llama-3-8B), you **must** use the base model's specific tokenizer.
* **BPE Training:** If building a model from scratch, train a Byte Pair Encoding (BPE) tokenizer on your specific corpus to capture domain-specific terminology.

---

### 5. Quality Filtering (The "Golden" Step)
For SLMs, **Quality > Quantity**.
* **Language Detection:** Prune any non-target language content.
* **Heuristic Filters:** Remove chunks with a high "symbol-to-alpha" ratio (usually indicative of broken tables or math formulas).
* **Perplexity Filtering:** Use a small pre-trained model to score text; discard high-perplexity chunks that likely contain gibberish.

---
# 📊 Part 2: Understanding Perplexity (PPL)

Perplexity measures how well a probability model predicts a sample. In NLP, it represents the **"effective branching factor"** of a model's vocabulary.



### Value Reference Table

| Value Range | Meaning | Significance |
| :--- | :--- | :--- |
| **1.0** | **Perfect Certainty** | Model is 100% sure; likely overfitted. |
| **10 – 30** | **High Performance** | Typical range for SOTA models (GPT-4, Llama 3). |
| **50 – 100** | **Moderate Fluency** | Basic logic is present; nuances are often lost. |
| **300+** | **High Confusion** | Model output is likely "word salad" or gibberish. |

---

### Key Mathematical Definition

Perplexity is the exponentiated average negative log-likelihood. It is defined as:

$$PPL(X) = 2^{H(X)}$$

Where $H(X)$ is the **Shannon entropy** of the distribution. 

> **Note:** A lower perplexity indicates a more "confident" model that is less surprised by the test data. However, low perplexity does not always equate to factual accuracy; a model can be confidently wrong (hallucinating).
# Part-3: Output Format
For compatibility with training frameworks like `Hugging Face trl` or `Axolotl`, the final data should be exported as a **JSONL** file:

```json
{"text": "The executive summary of the 2024 report states that... <|endoftext|>"}
{"text": "Section 2.1: Revenue growth was driven by... <|endoftext|>"}




