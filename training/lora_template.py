# Model-specific template: choose the exact VLM before running.
from transformers import AutoProcessor,AutoModelForVision2Seq,TrainingArguments
from peft import LoraConfig,get_peft_model

MODEL_NAME="REPLACE_WITH_SELECTED_VLM"

processor=AutoProcessor.from_pretrained(MODEL_NAME)
model=AutoModelForVision2Seq.from_pretrained(MODEL_NAME)

lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",
                target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
model=get_peft_model(model,lora)
model.print_trainable_parameters()

args=TrainingArguments(output_dir="./satquery_adapter",
                       num_train_epochs=3,
                       per_device_train_batch_size=1,
                       gradient_accumulation_steps=8,
                       learning_rate=2e-4,
                       logging_steps=20,
                       save_steps=250,
                       report_to="none")
print("Template initialized. The processor/collator/trainer must be customized to the selected VLM.")
