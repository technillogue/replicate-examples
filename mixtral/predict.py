import inspect
import json
import mmap
import os
import subprocess
import time
from typing import Iterator

from cog import BasePredictor, ConcatenateIterator, Input

from src import Llama, Message
from src.webrtc import RTC
from utils import maybe_download_with_pget, get_loop

MODEL_ID = "mixtral-8x7b-32kseqlen"
WEIGHTS_URL = "https://weights.replicate.delivery/hf/mixtral-8x7b-32kseqlen"
REMOTE_FILES = [
    "consolidated.00.pth-split00.pth",
    "consolidated.00.pth-split01.pth",
    "consolidated.00.pth-split02.pth",
    "consolidated.00.pth-split03.pth",
    "consolidated.00.pth-split04.pth",
    "consolidated.00.pth-split05.pth",
    "consolidated.00.pth-split06.pth",
    "consolidated.00.pth-split07.pth",
    "consolidated.00.pth-split08.pth",
    "params.json",
    "tokenizer.model",
]
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9


def consolidate_shards(prefix, output_file, num_shards, drop_shards=False):
    if os.path.exists(output_file):
        print("Output file already exists, skipping consolidation")
        return
    print(f"Consolidating {num_shards} shards into {output_file}")
    shard_files = [f"{prefix}-split{i:02d}.pth" for i in range(num_shards)]
    st = time.time()
    with open(output_file, "wb") as outfile:
        # this could be a lot better...
        subprocess.run(["cat"] + shard_files, stdout=outfile)
    if drop_shards:
        for shard in shard_files:
            os.remove(shard)
    print(f"concatenating shards took {time.time()-st:.3f}s")


# zipfile requires seekable
class SeekableMmap(mmap.mmap):
    def seekable(self) -> bool:
        return True


def load_weights() -> mmap.mmap:
    f = "weights-cache/consolidated.00.pth"
    if not os.path.exists(f):
        start = time.time()
        maybe_download_with_pget(
            path="weights-cache",
            remote_path=WEIGHTS_URL,
            remote_filenames=REMOTE_FILES,
        )
        print(f"downloading weights took {time.time() - start:.3f}s...now consolidating shards...")
        consolidate_shards(f, f, 9, drop_shards=False)
    return SeekableMmap(os.open(f, os.O_RDWR), 0)


class Predictor(BasePredictor):
    def setup(self):
        ckpt_mmap = load_weights()
        # consolidate_shards('weights-cache/consolidated.00.pth', 'weights-cache/consolidated.00.pth', 9)
        self.model = Llama.build(
            ckpt_dir="weights-cache",
            tokenizer_path="weights-cache/tokenizer.model",
            max_seq_len=32768,  # 2048 # 512
            max_batch_size=8,
            ckpt_filelike=ckpt_mmap,
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
            text = self.predict(**(self.defaults | args))

            # while True:
            #     # while-next() seems to express timing more clearly than for-in here
            #     tok_start = time.time()
            #     tok = next(stream, None)
            #     if tok is None:
            #         break
            now = time.time()

            #     token_count += 1
            yield {
                "text": text,
                "token_gen_latency": round((now - start) * 1000),
                # "gen_time": round((now - tok_start) * 1000),
                # "idx": token_count,
            }
            elapsed = time.time() - start
            tps = token_count / elapsed

            print(f"finished generating in {elapsed:.3f}, {tps:.2f} tok/s")
            yield {"status": "done"}  # , "tokens_per_second": round(tps, 3)}

        try:
            yield self.loop.run_until_complete(rtc.answer())
            yield self.loop.run_until_complete(rtc.wait_disconnect())
        except RuntimeError:
            self.loop = get_loop()
            yield self.loop.run_until_complete(rtc.answer())
            yield self.loop.run_until_complete(rtc.wait_disconnect())

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
        messages: str = Input(description="conversation messages as json", default="[]"),
        # TODO - support optionally returning logprobs
        # logprobs: bool = Input(
        #     description="Whether to return the log probabilities of the generated tokens.", default=False
        # ),
        webrtc_offer: str = Input(
            description="instead of a single prediction, handle a WebRTC offer as json, optionally with an ice_server key of ICE servers to use for connecting",
            default=None,
        ),
    ) -> str:
        if webrtc_offer:
            yield from self.webrtc_helper(webrtc_offer)
            return
        start = time.time()
        # just surface any JSONDecode error as-is
        message_list = json.loads(messages)
        if message_list:
            # i guess this isn't technically needed
            dialog = list(map(Message, message_list))
            results = self.model.chat_completion(
                dialogs=[dialog],
                max_gen_len=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                # logprobs=logprobs,
            )
            print(f"generation took {time.time() - start:.3f}s")
            return results[0]["generation"]["content"]  # results[0]["logprobs"]
        results = self.model.text_completion(
            prompts=[prompt],
            max_gen_len=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            # logprobs=logprobs,
        )
        print(f"generation took {time.time() - start:.3f}s")
        return results[0]["generation"]  # results[0]["logprobs"]

    # resolve Input to the correct default values
    defaults = {
        key: param.default.default
        for key, param in inspect.signature(predict).parameters.items()
        if hasattr(param.default, "default")
    }


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict(
        "Write a script to download the images for the top 10 posts of all time from /r/pics using the PRAW library.",
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
    )
    print(out)
