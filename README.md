# q14-lab-round001

A specialized inference engine for [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) at revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.5 LTS
- Reference: nxdi, tensor-parallel degree 2 (the engine gets as many accelerator cores)
- Judged under evaluation version 2
- Decode: not recorded (this run predates the one-forward-per-token rule)
- Source and data: https://github.com/xuanfeiren/enginesmith-q14-lab-round001
- Search: 7 rounds by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium), agent cost $361.26

## Notes from the author

- This run predates the held-out final evaluation and the one-forward-per-token rule (evaluation version 2), so the numbers below come from the search evaluation.
- Round 001 is the best round of this run that generates one token per forward pass. From round 002 on, the agent turned to speculative decoding, whose speed on the evaluation's random-token prompts overstates what it gives on real text, so those rounds (and their scores in the table below) are not what is shared here.
- A clean rebuild of round 001, checked later with the current procedure on fresh held-out prompts (5 rounds, the engine given only the reference's tensor-parallel degree in cores): passed, 4 of 2080 judged steps mismatch; score 1.76x NxD Inference (prefill 2.32x, decode 1.33x). Decode probe 1.03.

## Speed against nxdi (search evaluation)

Score 1.74x; prefill 2.32x, decode 1.31x (reference time / engine time, 65 tokens per request; the fraction of the hardware bound is bound time / engine time, an estimate).

| prompt tokens | prefill | decode | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 3.12x | 1.28x | not computed | not computed |
| 1,024 | 2.42x | 1.35x | not computed | not computed |
| 4,096 | 1.66x | 1.30x | not computed | not computed |

## Speed in absolute terms

Medians over the paired rounds of the search evaluation, this engine and nxdi on the same machine.

| prompt tokens | time to first token, ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 256 | 44.1 (nxdi 137.3) | 5,810.7 (1,864.3) | 45.1 (35.2) |
| 1,024 | 152.5 (nxdi 368.3) | 6,716.1 (2,780.2) | 44.7 (33.1) |
| 4,096 | 599.6 (nxdi 993.1) | 6,830.9 (4,124.4) | 43.1 (33.1) |

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.03x as long as one request for the same tokens at 256 prompt tokens (1,460.9 ms against 1,414.1 ms; each request 22.56 to 25.47 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went

- 8 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 2132 min.
- First correct engine: round 001, 271 min after the start.
- First engine as fast as nxdi (score 1.0): round 001, 271 min after the start.
- Agent usage: 2,368 input and 1,169k output tokens, 249,678k cache reads, $361.26 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 271.1 | 1.74x | 2.32x | 1.31x | 412 / 198k | 6,388 |
| 002 | optimize | accepted (score_improved) | 502.4 | 4.65x | 2.32x | 9.34x | 270 / 121k | 7,045 |
| 003 | optimize | accepted (score_improved) | 591.4 | 5.17x | 2.31x | 11.58x | 170 / 39k | 7,204 |
| 004 | optimize | accepted (score_improved) | 951.0 | 6.10x | 2.32x | 16.03x | 350 / 181k | 7,343 |
| 005 | optimize | accepted (score_improved) | 1,188.3 | 6.90x | 2.44x | 19.54x | 274 / 112k | 7,523 |
| 006 | optimize | accepted (score_improved) | 1,528.7 | 8.09x | 2.46x | 26.64x | 372 / 229k | 7,638 |
| 007 | optimize | rejected (insufficient_improvement) | 2,131.8 | 7.28x | 2.62x | 20.22x | 520 / 289k | 8,397 |

`history.csv` in https://github.com/xuanfeiren/enginesmith-q14-lab-round001 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local nxdi):

```sh
hf download Qwen/Qwen3-14B --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 --local-dir models/Qwen3-14B
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-q14-lab-round001 --model models/Qwen3-14B
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
