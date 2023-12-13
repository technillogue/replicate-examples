import inspect
import os
import time
from threading import Thread
from typing import Iterator

import torch
from cog import BasePredictor, ConcatenateIterator, Input
from utils import delay_prints, get_loop, maybe_download_with_pget
from webrtc import RTC


DEFAULT_MAX_NEW_TOKENS = os.environ.get("DEFAULT_MAX_NEW_TOKENS", 512)
DEFAULT_TEMPERATURE = os.environ.get("DEFAULT_TEMPERATURE", 0.6)
DEFAULT_TOP_P = os.environ.get("DEFAULT_TOP_P", 0.9)
DEFAULT_TOP_K = os.environ.get("DEFAULT_TOP_K", 50)

TORCH_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
DEVICE = "cuda"  # if torch.cuda.is_available() else "cpu"


class TextGenerationPredictor(BasePredictor):
    task = "text-generation"
    hf_model_id = None
    # prompt_template = "{prompt}"
    cache_dir = "./weights-cache"
    load_in_4bit = False
    use_safetensors = True
    generate_kwargs = {}
    local_files_only = True
    gcp_bucket_weights = None
    remote_filenames = None
    torch_dtype = "bf16"
    trust_remote_code = False

    def setup(self):
        start = time.time()
        maybe_download_with_pget(self.hf_model_id, self.gcp_bucket_weights, self.remote_filenames)
        print(f"downloading weights took {time.time() - start:.3f}s")

        # os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        global TextIteratorStreamer
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            TextIteratorStreamer,
        )

        config = AutoConfig.from_pretrained(
            self.hf_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        # resolve torch dtype from string.
        torch_dtype = TORCH_DTYPE_MAP[self.torch_dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_id,
            config=config,
            # quantization_config=bitsandbytes_config if self.load_in_4bit else None,
            torch_dtype=torch_dtype if not self.load_in_4bit else None,
            device_map="auto",
            cache_dir=self.cache_dir,
            use_safetensors=self.use_safetensors,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.loop = get_loop()

    def webrtc_helper(self, offer: str) -> Iterator[str]:
        print("creating connection for offer", offer)
        rtc = RTC(offer)

        @rtc.on_message
        def handler(args: dict) -> Iterator[dict[str, str | int | float]]:
            start = time.time()
            token_count = 0
            # that's right, this calls into predict recursively!
            # in principle you could create new/forwarded sessions from an existing connection?
            # resolve Input to the correct default values
            stream = self.predict(**(self.defaults | args))
            while True:
                # while-next() seems to express timing more clearly than for-in here
                tok_start = time.time()
                tok = next(stream, None)
                if tok is None:
                    break
                now = time.time()

                token_count += 1
                yield {
                    "text": tok,
                    "token_gen_latency": round((now - start) * 1000),
                    "gen_time": round((now - tok_start) * 1000),
                    "idx": token_count,
                }
            elapsed = time.time() - start
            tps = token_count / elapsed

            print(f"finished generating in {elapsed:.3f}, {tps:.2f} tok/s")
            yield {"status": "done", "tokens_per_second": round(tps, 3)}

        try:
            yield self.loop.run_until_complete(rtc.answer())
            yield self.loop.run_until_complete(rtc.wait_disconnect())
        except RuntimeError:
            self.loop = get_loop()
            yield self.loop.run_until_complete(rtc.answer())
            yield self.loop.run_until_complete(rtc.wait_disconnect())

    @delay_prints(REALLY_EAT_MY_PRINT_STATEMENTS=True)
    def predict(
        self,
        prompt: str,
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=DEFAULT_MAX_NEW_TOKENS,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.", default=DEFAULT_TEMPERATURE
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=DEFAULT_TOP_P,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=DEFAULT_TOP_K,
        ),
        prompt_template: str = Input(
            description="The template used to format the prompt before passing it to the model.",
            default="<s>[INST] {prompt} [/INST]",
        ),
        webrtc_offer: str = Input(
            description="instead of a single prediction, handle a WebRTC offer as json, optionally with an ice_server key of ICE servers to use for connecting",
            default=None,
        ),
    ) -> ConcatenateIterator:
        if webrtc_offer:
            yield from self.webrtc_helper(webrtc_offer)
            return
        prompt = prompt_template.format(prompt=prompt)
        inputs = self.tokenizer(
            [prompt], return_tensors="pt", add_special_tokens=False, return_token_type_ids=False
        ).to(DEVICE)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        for text in streamer:
            yield text


    # resolve Input to the correct default values
    defaults = {
        key: param.default.default
        for key, param in inspect.signature(predict).parameters.items()
        if hasattr(param.default, "default")
    }

Predictor = TextGenerationPredictor
Predictor.hf_model_id = "mixtral-8x7b-instruct-v0.1"
Predictor.gcp_bucket_weights = "https://weights.replicate.delivery/default/mixtral-8x7b-instruct-v0.1"
Predictor.remote_filenames = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    for text in p.predict(
        "Write a hello world FastAPI example",
        max_new_tokens=256,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
    ):
        print(text, end="")
