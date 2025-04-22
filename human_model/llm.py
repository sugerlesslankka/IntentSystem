from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    return model, tokenizer
    
def model_infer(tokenizer, model, prompt):
    conversation = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
        )
    inputs.to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response
