"""Compare separate baseline/profile runs by their saved greedy token arrays."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--mode", choices=("ar", "mtp"), default="ar")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least 2")
    if args.depth < 1:
        parser.error("--depth must be at least 1")

    import mlx.core as mx
    from mtplx.runtime import load
    from mtplx.generation import generate_ar, generate_mtpk, prefill_chunk_size_override
    from mtplx.sampling import SamplerConfig

    rt = load(args.model, mtp=args.mode == "mtp")
    prompts = [
        "Explain how a hash table works, including collisions, resizing, and time complexity.",
        "请详细解释数据库事务的 ACID 属性，并给出转账的例子。",
        "Write a Python LRU cache and explain how each method works.",
    ]
    def generate(prompt, count):
        ids = list(rt.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False))
        emitted = []
        kwargs = dict(max_tokens=count, sampler=SamplerConfig(temperature=0),
                      stop_token_ids=set(rt.tokenizer.eos_token_ids), token_callback=emitted.extend,
                      repetition_stop=False, loop_guard=False)
        with prefill_chunk_size_override(2048):
            if args.mode == "mtp":
                out = generate_mtpk(rt, ids, speculative_depth=args.depth,
                    verify_strategy="capture_commit", verify_core="stock", draft_core="stock",
                    mtp_cache_policy="persistent", mtp_history_policy="committed", **kwargs)
            else:
                out = generate_ar(rt, ids, **kwargs)
        visible = out.tokens[:-1] if out.tokens and out.tokens[-1] in rt.tokenizer.eos_token_ids else out.tokens
        assert emitted == visible, "streamed tokens differ from committed output"
        assert 0 < len(out.tokens) <= count
        stats = out.stats.to_dict()
        return dict(prompt_tokens=len(ids), tokens=out.tokens, finish_reason=out.finish_reason,
                    prefill_seconds=stats["prompt_eval_time_s"], decode_seconds=stats["decode_elapsed_s"],
                    decode_tps=stats["decode_tok_s"],
                    speculative={k: v for k, v in stats.items()
                                 if any(term in k for term in ("accept", "draft", "verify", "mtp"))})

    generate("Count from one to twenty.", 16)
    rows = [generate(prompt, args.tokens) for prompt in prompts]
    counters = {"island_calls": getattr(rt.model, "_island_calls", 0),
                "capture_island_calls": getattr(rt.model, "_capture_island_calls", 0),
                "capture_calls": sum(getattr(layer.linear_attn, "_capture_calls", 0)
                                     for layer in rt.model.layers if layer.is_linear)}
    if rt.local_optimization == "qwen35-mtp" and args.mode == "mtp":
        assert counters["capture_island_calls"] > 0
    if rt.local_optimization == "qwen35-ar":
        assert counters["island_calls"] > 0
    if rt.local_optimization in ("qwen27-mtp", "qwen35-mtp") and args.mode == "mtp":
        assert counters["capture_calls"] > 0
    report = dict(profile=rt.local_optimization, mode=args.mode,
                  depth=args.depth if args.mode == "mtp" else 0, rows=rows, counters=counters,
                  versions={name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
                  config_sha256=hashlib.sha256((args.model/"config.json").read_bytes()).hexdigest(),
                  peak_memory_gb=mx.get_peak_memory()/1e9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"profile": rt.local_optimization, "decode_tps": [r["decode_tps"] for r in rows],
                      "counters": counters}), flush=True)


if __name__ == "__main__":
    main()
