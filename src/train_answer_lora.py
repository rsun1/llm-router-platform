import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, Trainer
import torch 
from peft import LoraConfig, get_peft_model
from router_prompt import ANSWER_PROMPT_TEMPLATE
from datasets import Dataset


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DOMAIN_TYPE = 'legal'


outputname = f'answer_{DOMAIN_TYPE}_lora'
OUTPUT_LORA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),'output',outputname)


dataname= f'answer_{DOMAIN_TYPE}.csv'
data = pd.read_csv( os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train', dataname))

#data = pd.read_csv(TRAIN_PATH)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code = True)

def build_sample(query, label):
    prompt = ANSWER_PROMPT_TEMPLATE.format(query=query)
    token_query = tok(prompt,add_special_tokens=False)['input_ids']
    token_answer = tok(label, add_special_tokens=False)['input_ids']
    input_ids = token_query + token_answer + [tok.eos_token_id]
    answers = list(input_ids)
    n = len(token_query) 
    for i in range(n): 
        answers[i] = -100

    return {
        'input_ids': input_ids,
        'labels': answers
    }
    
if __name__ == '__main__':
    # sample = build_sample(data.loc[0,'query'], data.loc[0,'label'])
    # print(sum(1 for x in sample['labels'] if x != -100))
    # print(tok.decode([x for x in sample['labels'] if x != -100]))
    samples = []
    for q, lab in zip(data['query'],data['answer']):
        sample = build_sample(q, lab)
        samples.append(sample)
        
    dataset = Dataset.from_list(samples)
    print(len(dataset))
    print(dataset[0].keys())
    print(sum(1 for x in sample['labels']if x !=-100))
    
    # 1. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code = True,
        dtype = torch.float32,
    )    
    # 2. LoraConfig(照抄老师 21-28 行)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias = "none",
        task_type="CAUSAL_LM",    
    )
    # 3. model = get_peft_model(...)
    router_model = get_peft_model(base_model, lora_config)

    # 4. model.print_trainable_parameters()
    router_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_LORA_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=12,
        dataloader_num_workers=0,
        logging_steps=5,
        report_to = 'none',
        save_strategy='no',
    )
    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100)
    trainer = Trainer(
        model=router_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
        )
    trainer.train()
    router_model.save_pretrained(OUTPUT_LORA_DIR)
    tok.save_pretrained(OUTPUT_LORA_DIR)