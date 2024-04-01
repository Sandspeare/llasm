import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import pandas as pd
from llasm.model.llasm import LlasmLlamaForCausalLM
from llasm.conversation import conv_templates
from llasm.utils import disable_torch_init
from transformers import StoppingCriteria
from llasm.model.encoder import *
import numpy as np
import math
from llasm.conversation import conversation_lib

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_ASM_TOKEN = "<asm>"
DEFAULT_ASM_TOKEN_SITE = "<asm_site>"
DEFAULT_ASM_PATCH_TOKEN = "<asm_patch>"
DEFAULT_ASM_START_TOKEN = "<asm_start>"
DEFAULT_ASM_END_TOKEN = "<asm_end>"

def load_model():
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    if args.lora_enable:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.mm_projector is None:
        if args.lora_enable:
            print('Loading LLasm from base model...')
            llama_state_dict = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).state_dict()
            model = LlasmLlamaForCausalLM.from_pretrained(args.base_model_path, config=lora_cfg_pretrained, state_dict=llama_state_dict, torch_dtype=torch.float16, ignore_mismatched_sizes=True)

            print('Loading additional LLasm weights...')
            if os.path.exists(os.path.join(model_name, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_name, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_name, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.embed_tokens') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            non_lora_trainables = {k: v.to(torch.float16) for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_name)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Moving to CUDA...')
            model = model.cuda()
        else:
            model = LlasmLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        precessor = AsmTokenizer.from_pretrained(args.tokenizer)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_ASM_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_ASM_START_TOKEN, DEFAULT_ASM_END_TOKEN], special_tokens=True)

    else:
        model = LlasmLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        encoder = AsmEncoder.from_pretrained(args.encoder, torch_dtype=torch.float16).cuda()
        precessor = AsmTokenizer.from_pretrained(args.tokenizer, torch_dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_ASM_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_ASM_START_TOKEN, DEFAULT_ASM_END_TOKEN], special_tokens=True)

        encoder_config = encoder.config
        encoder_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ASM_PATCH_TOKEN])[0]
        encoder_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            encoder_config.im_start_token, encoder_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_ASM_START_TOKEN, DEFAULT_ASM_END_TOKEN])

        mm_projector = torch.nn.Linear(encoder_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.encoder = [encoder]
    model.get_tokenizer(tokenizer)
    return precessor, tokenizer, model

def query_name(asm, site, label):
    keywords = ['###']
    site = {}

    prompts = "Naming the given assembly function.\n<asm>."
    callsite = "\nThe function is called as \"target_function\" in the following.\n<asm_site>."

    if len(site) != 0:
        prompt = prompts + callsite
        tokens = precessor.encode_function(asm)
        asm_input = precessor.pad([tokens], padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        tokens = precessor.encode_function(site, 768)
        site_input = precessor.pad([tokens], padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        asm_input = {k: v.cuda() for k, v in asm_input.items()}
        site_input = {k: v.cuda() for k, v in site_input.items()}
        asm_token_len = asm_input['input_ids'].shape[1]
        site_token_len = site_input['input_ids'].shape[1]
    else:
        prompt = prompts
        tokens = precessor.encode_function(asm)
        asm_input = precessor.pad([tokens], padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False)
        asm_input = {k: v.cuda() for k, v in asm_input.items()}
        site_input = None
        asm_token_len = asm_input['input_ids'].shape[1]
        site_token_len = 0

    replace_token = DEFAULT_ASM_PATCH_TOKEN * asm_token_len
    replace_token = DEFAULT_ASM_START_TOKEN + replace_token + DEFAULT_ASM_END_TOKEN
    prompt = prompt.replace(DEFAULT_ASM_TOKEN, replace_token)
    
    if site_token_len != 0:
        replace_token = DEFAULT_ASM_PATCH_TOKEN * site_token_len
        replace_token = DEFAULT_ASM_START_TOKEN + replace_token + DEFAULT_ASM_END_TOKEN
        prompt = prompt.replace(DEFAULT_ASM_TOKEN_SITE, replace_token)

    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Human: " + prompt + ".\n### Assistant:"
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    model.get_rank(label)
    # new stopping implementation
    class KeywordsStoppingCriteria(StoppingCriteria):
        def __init__(self, keywords, tokenizer, input_ids):
            self.keywords = keywords
            self.tokenizer = tokenizer
            self.start_len = None
            self.input_ids = input_ids

        def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

            if self.start_len is None:
                self.start_len = self.input_ids.shape[1]
            else:
                outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                for keyword in self.keywords:
                    if keyword in outputs:
                        return True
            return False
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            assembles=[[asm_input, site_input]],
            do_sample=False,
            temperature=0.1,
            max_new_tokens=128,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

    while True:
        cur_len = len(outputs)
        outputs = outputs.strip()
        for pattern in ['###', 'Assistant:', 'Response:']:
            if outputs.startswith(pattern):
                outputs = outputs[len(pattern):].strip()
        if len(outputs) == cur_len:
            break

    try:
        index = outputs.index(keywords[0])
    except ValueError:
        outputs += keywords[0]
        index = outputs.index(keywords[0])
    outputs = outputs[:index].strip()
    return outputs

def load_supervisor():
    score_encoder = Supervisor.from_pretrained(args.encoder).cuda()
    supervisor = BertRegression(2048)
    supervisor.load_state_dict(torch.load(args.supervisor))
    return score_encoder, supervisor

def supervisor_score(asm, site):

    tokens = precessor.encode_function(asm)
    asm_input = precessor.pad([tokens], padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False).to(device)
    tokens = precessor.encode_function(site, 768)
    site_input = precessor.pad([tokens], padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False).to(device)

    asms = asms.to(device)
    asm_embeddings = score_encoder(**asm_input)
    site_embeddings = score_encoder(**site_input)
    score = supervisor(torch.cat((asm_embeddings, site_embeddings), dim=0)).squeeze()

    return score.item()

def normalized_l2_norm(vector, epsilon=1e-5):
    norm = np.linalg.norm(vector)
    normalized_norm = norm / np.sqrt(len(vector)) + epsilon
    return normalized_norm

def refresh_score(score, update_func, name):
    for func in binary:
        update = False
        for index in binary[func]["call_list"]:
            callee = binary[func]["call_list"][index]
            if callee == update_func:
                binary[func]["asm"][index] = binary[func]["asm"][index].replace(callee, name)
                update = True

        for index in binary[func]["site_list"]:
            callee = binary[func]["site_list"][index]
            if callee == update_func:
                binary[func]["site"][index] = binary[func]["site"][index].replace(callee, name) 
                update = True     
        if update == True:
            score[func] = supervisor_score(binary[func]["asm"], binary[func]["site"]) 
    return score

def analyze_binary():
    ans_file = open(os.path.join(args.save_path, args.name + ".json"), "w")
    score = {}
    for func in binary:
        if binary[func]["label"] == '':
            continue
        asm = binary[func]["asm"]
        site = binary[func]["site"]
        score[func] = supervisor_score(asm, site)
    
    while True:
        score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
        func = score.keys()[0]
        name = query_name(binary[func]["asm"], binary[func]["site"], binary[func]["label"])
        ans_file.write(json.dumps({ "score": round(normalized_l2_norm(model.rank) / 100, 4),
                                    "label": binary[func]["label"],
                                    "predict": name,
                                    "label_list": model.rank,
                                    "predict_list": model.predict_list
                                    }) + "\n")
        score.pop(func)
        ans_file.flush()
        if len(score) == 0:
            break
        score = refresh_score(score)
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # inference argument
    parser.add_argument("--model-name", type=str, default="../../models/llasm_finetune")
    parser.add_argument("--base-model-path", type=str, default="../../models/vicuna-13b-v1.5")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--lora_enable", type=bool, default=True)
    parser.add_argument("--device", type=int, default=0)
    # model path
    parser.add_argument("--supervisor", type=str, default="../../models/supervisor/supervisor.pth")
    parser.add_argument("--encoder", type=str, default="../../encoder/llasm/encoder")
    parser.add_argument("--tokenizer", type=str, default="../../encoder/llasm/tokenizer")
    # input and output
    parser.add_argument("--binary", type=str, default="mirai.json")
    parser.add_argument("--save-path", type=str, default="./save")
    parser.add_argument("--name", type=str, default="mirai")
    args = parser.parse_args([])

    torch.cuda.set_device(args.device)
    device = torch.device("cuda")

    binary = json.load(open(args.binary))
    ######################################################
    precessor, tokenizer, model = load_model()
    score_encoder, supervisor = load_supervisor()
    ######################################################
    analyze_binary()
    ######################################################
    json.dump(binary, fp=open(args.answers_file, "w"), indent=1)