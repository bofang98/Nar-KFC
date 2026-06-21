# Nar-KFC

Code release for the `Nar-KFC` project, organized as a cleaned evaluation toolkit fork for long-video understanding experiments.

This repository is based on `VLMEvalKit` and keeps the pieces needed to run the project code:

- evaluation entrypoints in `run.py` and `vlmeval/`
- benchmark helpers for video datasets such as `Video-MME`, `MLVU`, `LongVideoBench`, `MVBench`, and `TempCompass`
- example launch scripts in `command/`

Public-release cleanup applied to this repository:

- removed cached bytecode and `__pycache__`
- removed local evaluation outputs and temporary logs
- removed machine-specific environment `source` commands from example scripts
- replaced several hard-coded local model paths with public model identifiers

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

Example commands are provided in `command/`:

```bash
bash command/run_mlvu_qwenvl.sh
bash command/run_videomme_internvl2.sh
bash command/run_longvideobench_llava_video.sh
```

You will likely need to adjust:

- `CUDA_VISIBLE_DEVICES`
- model names in the scripts
- dataset cache locations or relevant environment variables

## Notes

- Benchmark outputs are intentionally not tracked in the repository.
- Some datasets require manual download or access approval from their original sources.
- If a model or dataset is stored locally on your machine, update the corresponding config or environment variable before running.

## License

See [LICENSE](LICENSE).
