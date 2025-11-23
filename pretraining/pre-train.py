from datasets import load_dataset
import os
from transformers import(
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCasualLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(examples):
    return tokenizer(examples["text"])

if __name__ == "__main__":
    dataset = load_dataset("text", data_files=r"D:\genaii\assignment\combined_dataset(q2).txt")

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched = True,
        num_proc=2,
        remove_columns=dataset["train"].column_names
    )

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
    )

    model = LlamaForCasualLM(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./llama2-mini-model',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_steps=500,
        save_steps=100,
        save_total_limit=2,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained('./llama2-mini-model')
    tokenizer.save_pretrained('./llama2-mini-model')

    from transformers import pipeline
    generator = pipeline('text-generation', model='./llama2-mini-model', tokenizer='./llama2-mini-model')

    prompt = "what is cricket"
    output = generator(prompt, max_length=100, num_return_sequences=1)
    print(output[0]['generated_text'])
