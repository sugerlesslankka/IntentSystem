from human_model.loader import load_model_and_processor
from human_model.utils import get_visual_type, VALID_DATA_FORMAT_STRING
import os
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import cv2
from scene_model.utils.misc import generate2_adpt_if_nodist
from scene_model.project.models.model import MappingType, CaptionModel
from transformers import GPT2Tokenizer
from scene_model.clip1.clip import _transform
from PIL import Image
import torch.distributed as dist
import time
from random import sample

# dist.init_process_group("nccl", init_method='file:///tmp/somefile', rank=0, world_size=1)
# torch.backends.cudnn.enabled = False
human_time = []
scene_time = []
reason_time = []

def inference_human(model, processor, prompt, video_file, generate_kwargs):
    start_time = time.time()
    try:
        inputs = processor(prompt, video_file, edit_prompt=True, return_prompt=True)
    except:
        print('corrupted:', video_file)
        return None
    if 'prompt' in inputs:
        inputs.pop('prompt')
    inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )
    output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    end_time = time.time()
    print('完成人像处理，用时'+str(end_time-start_time)+'秒')
    human_time.append(end_time-start_time)
    return output_text

def inference_scene(model, video_file, kw, tokenizer):
    cap = cv2.VideoCapture(video_file)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames * 3 // 4 - 1)
    success, image = cap.read()
    start_time = time.time()
    image = Image.fromarray(image)
    image = _transform(224)(image)
    image = image.cuda(non_blocking=True)
    kt = torch.tensor(tokenizer.encode(kw))
    padding = 20 - kt.shape[0]
    if padding > 0:
        kt = torch.cat((kt, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        kt = kt[:20]
    mask_kt = kt.ge(0)
    kt[~mask_kt] = 0
    kt = kt.cuda(non_blocking=True)
    image = image.unsqueeze(0)
    kt = kt.unsqueeze(0)
    prefix, len_cls = model.image_encode(image)
    prefix_embed = model.clip_project(prefix)
    kt = model.gpt.transformer.wte(kt)
    len_pre = model.len_head(len_cls)
    prefix_embed = model.kw_att(prefix_embed, kt)
    generated_text_prefix = generate2_adpt_if_nodist(model, tokenizer, embed=prefix_embed, len_pre=len_pre.argmax(-1) + 1)
    end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()
    print('完成场景处理，用时'+str(end_time-start_time)+'秒')
    scene_time.append(end_time-start_time)
    return generated_text_prefix

def inference_llm(llm, llm_tokenizer, prompt):
    start_time = time.time()
    messages = [{"role": "user", "content": prompt}]
    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    llm_inputs = llm_tokenizer([text], return_tensors="pt").to(llm.device)
    generated_ids = llm.generate(
        llm_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(llm_inputs.input_ids, generated_ids)
    ]
    output_text = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end_time = time.time()
    print('完成推理，用时'+str(end_time-start_time)+'秒')
    reason_time.append(end_time-start_time)
    return output_text

def form_prompt(human_pred, scene_pred):
    prompt = f"""你是一个协作机器人，从电脑的前置摄像头观察，负责协助一名软件工程师，拥有进行以下指令的能力：
指令1：按摩。只要观察到协作人十分疲惫，比如睡着了的感觉，请执行这一指令。
指令2：打扫。只要观察到协作人离开座位了或者看手机了，请执行这一指令。
指令3：不进行任何操作，如果没有观察到任何特殊情况，请执行这一指令。 
以下英文描述代表协作人的人物描述和当前时刻的场景大致状态，需要从中判断你要执行什么指令，请你先仔细翻译和读取当前协作人的详细英文描述，最后输出需要执行指令对应的序号：
人物详细描述：{human_pred}
场景大致描述：{scene_pred}
输出格式要求：
这个人的详细行为和意图总结（用中文）：XXXXXXXXXXXX
对应的指令：X（X为1，2或3）
"""
    return prompt 


def run():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="robot", help='Path to video/image; or Dir to videos/images')
    parser.add_argument("--max_samples", type=int, default=0, help="Limit sample num.")
    parser.add_argument('--output_path', type=str, default="./robot_test_2.json", help='Path to save result, has to be a json file')
    args = parser.parse_args()
    # 查看路径
    assert os.path.exists(args.input_path), f"input_path not exist: {args.input_path}"
    if os.path.isdir(args.input_path):
        input_files = [os.path.join(args.input_path, file) for file in os.listdir(args.input_path)]
        if args.max_samples > 0:
            input_files = input_files[:args.max_samples]
    else:
        input_files = [args.input_path]
    assert len(input_files) > 0, f"None valid input file in: {args.input_path} {VALID_DATA_FORMAT_STRING}"
    
    assert os.path.exists(os.path.dirname(args.output_path)), f"output_path not exist: {args.output_path}"
    assert args.output_path.endswith('.json'), f"invalid json file name: {args.output_path}"
    # 读取human模型
    human_model, processor = load_model_and_processor('PEAR-7b', max_n_frames=8, attn_implementation="flash_attention_2")
    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": 512,
        "top_p": 1,
        "temperature": 0,
        "use_cache": True
    }
    # 读取scene模型
    scene_model = CaptionModel(10, clip_length=10, prefix_size=512,
                                 num_layers=8, mapping_type=MappingType.MLP, Timestep=20,
                                 if_drop_rate=0.1)
    scene_model.load_state_dict(torch.load('KCDN-small/ckpt.pt', map_location=torch.device('cpu'))["model"])
    scene_model = scene_model.cuda()
    scene_model.eval()
    scene_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # 读取llm
    llm_name = 'qwen1.5'
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype="auto", device_map="auto")
    
    intent_data = []
    for input_file in tqdm(input_files, desc="Generating..."):
        visual_type = get_visual_type(input_file)
        prompt = " USER: Please describe:\n1. The person's appearance, wearing, and other important charateristic.\n2. The person's each action in order and the objects the person interacted with.\n3. The person's mood, feeling, emotion or intention if can be detected.</s> ASSISTANT: "
        if visual_type == 'video':
            prompt = "<video>\n" + prompt.replace("<image>", "").replace("<video>", "")
        else:
            prompt = "<image>\n" + prompt.replace("<image>", "").replace("<video>", "")
        keyword = ","
        human_pred = inference_human(human_model, processor, prompt, input_file, generate_kwargs)
        scene_pred = inference_scene(scene_model, input_file, keyword, scene_tokenizer)
        # scene_pred = ''
        prompt = form_prompt(human_pred, scene_pred)
        analysis = inference_llm(llm, llm_tokenizer, prompt)
        intent_data.append({'video_path':input_file, 'human':human_pred, 'scene':scene_pred, 'analysis':analysis})
    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(intent_data, json_file, ensure_ascii=False, indent=4)
    print(sum(human_time)/len(human_time))
    print(sum(scene_time)/len(scene_time))
    print(sum(reason_time)/len(reason_time))

if __name__ == "__main__":
    run()