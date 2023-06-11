import copy
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import modules.shared as shared
from modules.callbacks import Iteratorize

np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' if shared.args.rwkv_cuda_on else '0'  

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class RWKVModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path, dtype="fp16", device="cuda"):
        tokenizer_path = Path(f"{path.parent}/20B_tokenizer.json")
        if shared.args.rwkv_strategy is None:
            model = RWKV(model=str(path), strategy=f'{device} {dtype}')
        else:
            model = RWKV(model=str(path), strategy=shared.args.rwkv_strategy)

        pipeline = PIPELINE(model, str(tokenizer_path))
        result = self()
        result.pipeline = pipeline
        result.model = model
        result.cached_context = ""
        result.cached_model_state = None
        result.cached_output_logits = None
        return result

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=None, alpha_frequency=0.1, alpha_presence=0.1, token_ban=None, token_stop=None, callback=None):
        args = PIPELINE_ARGS(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alpha_frequency=alpha_frequency,  
            alpha_presence=alpha_presence,  
            token_ban=token_ban or [0], 
            token_stop=token_stop or []
        )

        if self.cached_context != "":
            if context.startswith(self.cached_context):
                context = context[len(self.cached_context):]
            else:
                self.cached_context = ""
                self.cached_model_state = None
                self.cached_output_logits = None

      
        out = self.generate_from_cached_state(context, token_count=token_count, args=args, callback=callback)
        return out

    def generate_with_streaming(self, **kwargs):
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply

    
    def generate_from_cached_state(self, ctx="", token_count=20, args=None, callback=None):
        all_tokens = []
        out_str = ''
        occurrence = {}
        state = copy.deepcopy(self.cached_model_state) if self.cached_model_state is not None else None

       
        if ctx == "":
            out = self.cached_output_logits

        for i in range(token_count):
            
            tokens = self.pipeline.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]
            if i == 0:
                begin_token= len(all_tokens)
                last_token_posi=begin_token
            


            if i == 0:
                self.cached_context += ctx
                self.cached_model_state = copy.deepcopy(state)
                self.cached_output_logits = copy.deepcopy(out)

           
            for n in args.token_ban:
                out[n] = -float('inf')

            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

          
            token = self.pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break

            all_tokens += [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

           
            tmp = self.pipeline.decode(all_tokens[last_token_posi:])
            if '\ufffd' not in tmp:  
                if callback:
                    callback(tmp)
                    
                out_str += tmp
                last_token_posi = begin_token + i + 1
        return out_str


class RWKVTokenizer:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path):
        tokenizer_path = path / "20B_tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        result = self()
        result.tokenizer = tokenizer
        return result

    def encode(self, prompt):
        return self.tokenizer.encode(prompt).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
