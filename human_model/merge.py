
from models.loader import load_model_and_processor
from peft import PeftModel

# 加载基础模型
base_model_name = "/home/panmx/IntentRecognition/checkpoints/r3_pos_model"
base_model, processor = load_model_and_processor(base_model_name, max_n_frames=8)

# 加载 LoRA 微调的权重
lora_model_path = "/home/panmx/IntentRecognition/checkpoints/r3"
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# 将 LoRA 权重合并到基础模型中
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("/home/panmx/IntentRecognition/checkpoints/r3_model")
processor.processor.processor.save_pretrained("/home/panmx/IntentRecognition/checkpoints/r3_model")