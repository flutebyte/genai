from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        tokenizer_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        tokenizer_inputs["labels"] = tokenizer_inputs["input_ids"].copy()
        return tokenizer_inputs


    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2 
    )
    return tokenized

def main():
    dataset = load_dataset("text", data_files="assignment1/dr_patient.txt")  

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)  

    # loraa
    lora_config = LoraConfig(
        r=16,  # r controls the no. of features lora will learn
        lora_alpha=32,  # lora alpha decides how strongly to apply those features
        target_modules=["c_attn"],  # this is attention layer that turns matrix to query key and value
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = GPT2LMHeadModel.from_pretrained("gpt2")  # loading base model
    peft_model = get_peft_model(model, lora_config)  # apply lora

    training_args = TrainingArguments(
        output_dir="lora_model",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        overwrite_output_dir=True,  # overwrite if folder exists
        save_steps=500,
        save_total_limit=2
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],  # training on tokenized dataset
    )

    trainer.train()  # start training
    peft_model.save_pretrained("lora_model")  # save LoRA model

    input_prompt = "Doctor : I have headache"

    from transformers import pipeline

    generator = pipeline("text-generation", model="./lora_model", tokenizer=tokenizer)

    result = generator(input_prompt, max_length=50, do_sample=True)  # generate response
    print(result[0]["generated_text"])

if __name__ == "__main__":
    main()
