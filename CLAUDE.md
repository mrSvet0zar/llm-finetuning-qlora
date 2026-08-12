# 🤖 CLAUDE.md - Projet 2: Fine-Tuning & Custom LLM

## 📌 Objectif du Projet
Fine-tuner un modèle open-source sur un dataset spécialisé pour démontrer la maîtrise du transfer learning, optimisation hyperparamètres, et entraînement de modèles. Déployer le modèle pour inférence et créer une API.

---

## 🛠️ Stack Technologique Complète

### Training Environment
- **Python 3.10+**
- **PyTorch 2.0+**
- **Hugging Face Transformers** (dernière version)
- **PEFT** (Parameter-Efficient Fine-Tuning) pour LoRA
- **bitsandbytes** (quantization 4-bit)
- **Accelerate** (distributed training)
- **Wandb** (experiment tracking)
- **Datasets** (Hugging Face)

### Inference & Serving
- **Ollama** (local inference) OU
- **Hugging Face Inference API** OU
- **FastAPI** (custom server)
- **Transformers pipeline** (simple inference)

### Infrastructure
- **Google Colab** (GPU gratuit: T4/A100) OU
- **Vast.ai** (GPU pas cher) OU
- **Local machine** (si GPU 8GB+)
- **Hugging Face Hub** (model hosting)

### Deployment
- **Hugging Face Hub** (model repository)
- **Ollama** (local, quantized)
- **FastAPI + Railway** (production API)

---

## 📐 Architecture d'Entraînement

```
┌────────────────────────────────────────────────┐
│           Raw Dataset (Q&A Pairs)              │
└────────────┬─────────────────────────────────┘
             │
        ┌────▼─────────────────┐
        │ Data Preprocessing   │
        │ - Cleaning           │
        │ - Formatting JSONL   │
        │ - Split train/val    │
        └────┬─────────────────┘
             │
        ┌────▼────────────────────┐
        │ Select Base Model       │
        │ (Mistral-7B recommended)│
        └────┬────────────────────┘
             │
        ┌────▼──────────────────────┐
        │ Setup LoRA Adapters       │
        │ - rank=8                  │
        │ - alpha=16                │
        │ - target modules          │
        └────┬──────────────────────┘
             │
        ┌────▼────────────────────┐
        │ Training Loop            │
        │ - 3-5 epochs             │
        │ - Learning rate: 2e-4    │
        │ - Batch size: 4          │
        └────┬────────────────────┘
             │
        ┌────▼──────────────────┐
        │ Evaluation             │
        │ - BLEU score          │
        │ - ROUGE metrics       │
        │ - Custom metrics      │
        └────┬──────────────────┘
             │
        ┌────▼───────────────────┐
        │ Merge & Quantize       │
        │ - Merge LoRA weights   │
        │ - Convert to GGML      │
        └────┬───────────────────┘
             │
        ┌────▼──────────────────┐
        │ Deployment            │
        │ - HF Hub              │
        │ - Ollama              │
        │ - FastAPI             │
        └──────────────────────┘
```

---

## 📋 Phase 1: Dataset Preparation

### 1.1 Dataset Collection

**Cas d'usage recommandé: Domain Expert Q&A**

Structure du dataset:
```json
{
  "instruction": "Qu'est-ce que la vectorization en Python?",
  "input": "",
  "output": "La vectorization est l'optimisation des opérations NumPy/Pandas pour travailler sur des arrays entiers plutôt que des boucles..."
}
```

**Sources de données:**
- Documentation technique personnelle
- Stack Overflow questions/answers (scraping légal)
- Blog posts et articles
- Books ou papers
- Support tickets/FAQ

**Nombre d'exemples requis:** 500-2000 pairs Q&A

### 1.2 Data Preparation Script

**File: `prepare_dataset.py`**
```python
import json
from typing import List, Dict
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def load_raw_data(source_file: str) -> List[Dict]:
    """Load raw Q&A data from various sources"""
    with open(source_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_qa_pair(instruction: str, output: str, input_text: str = "") -> str:
    """Format Q&A pair using standard template"""
    template = f"<s>[INST] {instruction}"
    if input_text:
        template += f"\n{input_text}"
    template += f" [/INST] {output} </s>"
    return template

def validate_dataset(data: List[Dict]) -> List[Dict]:
    """Validate and clean dataset"""
    validated = []
    
    for item in data:
        # Check required fields
        if not item.get("instruction") or not item.get("output"):
            print(f"Skipping invalid item: {item}")
            continue
        
        # Check length
        if len(item["instruction"]) < 10 or len(item["output"]) < 20:
            print(f"Skipping too short item: {item}")
            continue
        
        # Check for duplicates (simple check)
        if not any(v["instruction"] == item["instruction"] for v in validated):
            validated.append(item)
    
    print(f"Validated {len(validated)} items out of {len(data)}")
    return validated

def create_training_dataset(
    data: List[Dict],
    output_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """Create JSONL training/validation datasets"""
    
    # Shuffle
    random.shuffle(data)
    
    # Split
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Write JSONL
    for split, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = output_file.replace(".jsonl", f"_{split}.jsonl")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                formatted = format_qa_pair(
                    item["instruction"],
                    item["output"],
                    item.get("input", "")
                )
                f.write(json.dumps({"text": formatted}) + "\n")
        
        print(f"Created {split} set: {output_path} ({len(split_data)} items)")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total: {len(data)} items")
    print(f"Train: {len(train_data)} ({train_ratio*100:.0f}%)")
    print(f"Val: {len(val_data)} ({val_ratio*100:.0f}%)")
    print(f"Test: {len(test_data)} ({(1-train_ratio-val_ratio)*100:.0f}%)")

if __name__ == "__main__":
    # Load data
    raw_data = load_raw_data("raw_qa_data.json")
    
    # Validate
    validated_data = validate_dataset(raw_data)
    
    # Create training sets
    create_training_dataset(validated_data, "dataset.jsonl")
```

### 1.3 Dataset File Format

**File: `dataset_train.jsonl`**
```json
{"text": "<s>[INST] What is Python vectorization? [/INST] Vectorization is optimizing operations using NumPy/Pandas arrays instead of loops... </s>"}
{"text": "<s>[INST] Explain async/await in Python [/INST] Async/await enables asynchronous programming by allowing functions to be paused... </s>"}
...
```

---

## 📋 Phase 2: Environment Setup

### 2.1 Google Colab Setup

```python
# In first cell of Colab notebook

# Install dependencies
!pip install -q torch transformers peft bitsandbytes accelerate datasets wandb huggingface-hub
!huggingface-cli login  # Login to Hugging Face

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set HuggingFace cache
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/huggingface_cache"
```

### 2.2 Local Setup (Alternative)

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# requirements.txt content:
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
datasets>=2.14.0
wandb>=0.15.0
huggingface-hub>=0.19.0
sentencepiece
protobuf
```

---

## 📋 Phase 3: Fine-Tuning Implementation

### 3.1 Complete Training Script

**File: `train.py`**
```python
import os
import torch
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model
import wandb

# Configuration
@dataclass
class Config:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    dataset_path: str = "dataset_train.jsonl"
    output_dir: str = "./fine-tuned-model"
    
    # LoRA config
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = None
    
    # Training config
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048
    
    # Infrastructure
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    device_map: str = "auto"
    seed: int = 42

def setup_quantization_config(config: Config) -> Optional[BitsAndBytesConfig]:
    """Setup 4-bit quantization"""
    if not config.use_4bit:
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

def load_model_and_tokenizer(config: Config):
    """Load model and tokenizer with quantization"""
    
    # Quantization config
    bnb_config = setup_quantization_config(config)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=config.device_map,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora(model, config: Config):
    """Setup LoRA adapters"""
    
    if config.lora_target_modules is None:
        # Auto-detect target modules based on model
        if "mistral" in config.model_name.lower():
            config.lora_target_modules = ["q_proj", "v_proj"]
        elif "llama" in config.model_name.lower():
            config.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            config.lora_target_modules = ["q_proj", "v_proj"]
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def prepare_dataset(config: Config, tokenizer):
    """Load and tokenize dataset"""
    
    # Load dataset
    dataset = load_dataset("json", data_files=config.dataset_path)
    
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length"
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )
    
    # Split into train and validation
    train_val = tokenized_dataset["train"].train_test_split(test_size=0.1)
    
    return train_val["train"], train_val["test"]

def main():
    # Setup
    config = Config()
    set_seed(config.seed)
    
    # Initialize Wandb
    wandb.init(
        project="fine-tuning",
        name=f"{config.model_name.split('/')[-1]}-lora",
        config=config.__dict__
    )
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup LoRA
    print("Setting up LoRA...")
    model = setup_lora(model, config)
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        optim="paged_adamw_8bit",
        report_to=["wandb"],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Push to Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    trainer.model.push_to_hub("your-username/model-name")
    tokenizer.push_to_hub("your-username/model-name")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
```

### 3.2 Inference Script

**File: `inference.py`**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

def load_fine_tuned_model(model_path: str):
    """Load fine-tuned model"""
    
    # Load from local path
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 256) -> str:
    """Generate response"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Load model
    model, tokenizer = load_fine_tuned_model("./fine-tuned-model")
    
    # Test prompts
    prompts = [
        "What is vectorization in Python?",
        "Explain async/await",
        "How to optimize database queries?"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_response(model, tokenizer, prompt)
        print(f"Response: {response}")
```

---

## 📋 Phase 4: Evaluation

### 4.1 Evaluation Script

**File: `evaluate.py`**
```python
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def evaluate_model(model_path: str, test_file: str):
    """Evaluate fine-tuned model"""
    
    # Load model
    model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load test data
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Metrics
    rouge_scores = []
    bleu_scores = []
    
    for item in test_data[:100]:  # Evaluate on 100 samples
        # Extract instruction and expected output
        text = item["text"]
        instruction = text.split("[INST]")[1].split("[/INST]")[0].strip()
        expected = text.split("[/INST]")[1].split("</s>")[0].strip()
        
        # Generate
        prompt = f"<s>[INST] {instruction} [/INST]"
        generated = generate_response(model, tokenizer, prompt)
        
        # Calculate ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        rouge_score = scorer.score(expected, generated)
        rouge_scores.append(rouge_score['rouge1'].fmeasure)
        
        # Calculate BLEU
        expected_tokens = expected.split()
        generated_tokens = generated.split()
        bleu_score = sentence_bleu([expected_tokens], generated_tokens)
        bleu_scores.append(bleu_score)
    
    # Print results
    print(f"Average ROUGE: {np.mean(rouge_scores):.4f}")
    print(f"Average BLEU: {np.mean(bleu_scores):.4f}")
    
    return {
        "avg_rouge": float(np.mean(rouge_scores)),
        "avg_bleu": float(np.mean(bleu_scores))
    }

if __name__ == "__main__":
    metrics = evaluate_model("./fine-tuned-model", "dataset_test.jsonl")
    print(f"\nMetrics: {metrics}")
```

---

## 📋 Phase 5: Model Merging & Quantization

### 5.1 Merge LoRA Weights

**File: `merge_model.py`**
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_lora_weights(model_path: str, output_path: str):
    """Merge LoRA weights into base model"""
    
    # Load LoRA model
    model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    
    # Merge
    model = model.merge_and_unload()
    
    # Save
    model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Model merged and saved to {output_path}")

if __name__ == "__main__":
    merge_lora_weights("./fine-tuned-model", "./merged-model")
```

### 5.2 Convert to GGML (For Ollama)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Convert to GGML
python convert.py merged-model --outfile merged-model.ggml

# Quantize
./quantize merged-model.ggml merged-model.q4_K_M.ggml Q4_K_M
```

---

## 📋 Phase 6: Deployment

### 6.1 Hugging Face Hub (Recommended)

```python
# Push model to Hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./merged-model",
    repo_id="your-username/model-name",
    repo_type="model"
)
```

### 6.2 Ollama Deployment

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./merged-model.q4_K_M.ggml
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create model
ollama create my-finetuned-model -f Modelfile

# Run
ollama run my-finetuned-model "Your prompt here"
```

### 6.3 FastAPI Server

**File: `api_server.py`**
```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model once on startup
model = None
tokenizer = None

@app.on_event("startup")
async def startup():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained("./fine-tuned-model")
    tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 256
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    prompt: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=request.max_length,
            temperature=request.temperature,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return GenerateResponse(text=text, prompt=request.prompt)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## ✅ Checklist de Développement

- [ ] Dataset collected & validated
- [ ] Data split into train/val/test
- [ ] JSONL format files created
- [ ] Google Colab notebook prepared
- [ ] Training script functional
- [ ] Model training completed (3-5 epochs)
- [ ] Evaluation metrics calculated
- [ ] Model merged (LoRA weights)
- [ ] Model quantized (GGML)
- [ ] Model pushed to Hugging Face Hub
- [ ] Ollama model created & tested
- [ ] FastAPI inference server working
- [ ] Dockerfile for deployment
- [ ] Performance benchmarks
- [ ] Blog post documenting process

---

## 🎯 Critères de Succès

✅ **Performance:**
- BLEU score >= 20
- ROUGE score >= 0.30
- Inference latency < 2s (for 100 tokens)

✅ **Model Quality:**
- Coherent responses on test prompts
- Maintains domain expertise
- No hallucinations on simple queries

✅ **Reproducibility:**
- All code in GitHub
- Dataset (anonymized) published
- Training logs saved
- Hyperparameters documented

✅ **Deployment:**
- Model on Hugging Face Hub
- Ollama model working
- FastAPI endpoint live
- Demo notebook functional

---

## 📊 Expected Training Time

- **Google Colab T4:** 4-6 hours per epoch (12-18h total)
- **Google Colab A100:** 2-3 hours per epoch (6-9h total)
- **RTX 3090:** 1-2 hours per epoch (3-6h total)

---

## 🚀 Next Steps (Post-MVP)

- [ ] Increase dataset size (2000+ examples)
- [ ] Multi-turn conversation fine-tuning
- [ ] Instruction following improvements
- [ ] Benchmark against baseline
- [ ] Create paper/blog post
- [ ] Open-source release

---

**📅 Timeline estimée: 3-4 semaines** (mostly training time)

Bonne chance! 🎯
