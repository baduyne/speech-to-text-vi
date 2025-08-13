import os
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from peft import LoraConfig
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

# Hugging Face login
def hf_login():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN not found in .env file")
    login(hf_token)
    print("Logged in to Hugging Face")


# Dataset loading & preprocessing
def load_and_prepare_dataset():
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "vi", trust_remote_code=True)
    splits_to_merge = ["train", "validation", "other", "validated"]
    datasets_to_merge = [dataset[split] for split in splits_to_merge if split in dataset]
    full_dataset = concatenate_datasets(datasets_to_merge).shuffle(seed=3107)
    print(f"Total samples: {len(full_dataset)}")

    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=3107)

    train_dataset = split_dataset["train"].select_columns(["audio", "sentence"])
    valid_dataset = split_dataset["test"].select_columns(["audio", "sentence"])

    # Cast audio column
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    return train_dataset, valid_dataset


# Load processors (feature extractor + tokenizer + processor)
def load_processors():
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Vietnamese", task="transcribe")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="vi", task="transcribe")
    return feature_extractor, tokenizer, processor



# Prepare dataset with audio features and tokenized labels
def prepare_dataset(dataset, feature_extractor, tokenizer, max_length=128, num_proc=4):
    def encode(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        encoded_labels = tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        batch["labels"] = encoded_labels.input_ids.squeeze(0)
        return batch

    return dataset.map(encode, remove_columns=["audio", "sentence"], num_proc=num_proc)

# Metric
def compute_metrics(pred, tokenizer):
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input_features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(dtype=torch.float16)

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

# Build model
def build_model(device="cpu"):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if device == "cuda" else None
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"

    # LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model.add_adapter(lora_config, adapter_name="lora_1")
    return model


# Training arguments
def get_training_args(output_dir="./whisper-small-vi", num_epochs=15):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=14,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=100,
        num_train_epochs=num_epochs,                                        
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=14,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )



# Main training function
def main():
    hf_login()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    train_ds, valid_ds = load_and_prepare_dataset()

    # Processors
    feat_ext, tokenizer, processor = load_processors()

    # Prepare datasets
    train_ds = prepare_dataset(train_ds, feat_ext, tokenizer)
    valid_ds = prepare_dataset(valid_ds, feat_ext, tokenizer)

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Model
    model = build_model(device)

    # Trainer
    training_args = get_training_args(num_epochs=15)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    model.config.use_cache = False

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
