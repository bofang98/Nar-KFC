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

## Environment Notes

Different models may require different runtime environments and dependency versions. In practice, you should prepare model-specific environments before running evaluation, instead of assuming that a single Python environment will work for every model.

Please adapt the environment according to the official `VLMEvalKit` requirements, especially for dependencies such as `transformers`, `torchvision`, and `flash-attn` when required by specific models. If a target model fails under the current environment, first check the corresponding version recommendations from the original `VLMEvalKit` repository and switch to a matching environment before evaluation.

## Sampling Implementation Notes

Taking `Video-MME` as an example, the KFC keyframe sampling call is:

```python
frames, indices, video_info, fps_1_frames_index = self.graph_video_frames(
    vid_clip_feat, query_clip_feat, line, num_frames, fps, video_llm, rank
)
```

The greedy frame selection logic is implemented in [`vlmeval/smp/vlm.py`](vlmeval/smp/vlm.py) under `def incremental_select_nodes(matrix, K, shrink_K, rank):`.

The Integer Quadratic Programming frame selection logic is implemented in [`vlmeval/smp/vlm.py`](vlmeval/smp/vlm.py) under `def ILP_select_nodes(matrix, K, shrink_K, rank):`.

Cleaned excerpts of the two core functions are shown below.

Greedy frame selection:

```python
def incremental_select_nodes(matrix, K, shrink_K, rank):
    device = matrix.device
    n = matrix.shape[0]
    sym_matrix = (matrix + matrix.T).to(device)

    if n <= 4 * K:
        U, Sigma, Vt = torch.svd(sym_matrix)
        Sigma_k = torch.diag(Sigma[: n // 4])
        U_k = U[:, : n // 4]
        Vt_k = Vt[:, : n // 4]
        S_k = U_k @ Sigma_k @ Vt_k.T
        start = torch.argmax(S_k.diagonal()).item()
        selected = {start}
        indices_mapping = {i: i for i in range(n)}
    else:
        if n > 7200:
            S_k = sym_matrix
        else:
            U, Sigma, Vt = torch.svd(sym_matrix)
            Sigma_k = torch.diag(Sigma[: n // 4])
            U_k = U[:, : n // 4]
            Vt_k = Vt[:, : n // 4]
            S_k = U_k @ Sigma_k @ Vt_k.T

        sparse_matrix, _, indices_mapping = uniform_select_matrix(S_k, M=shrink_K)
        start = torch.argmax(sparse_matrix.diagonal()).item()
        selected = {start}
        sym_matrix = sparse_matrix

    sym_matrix.fill_diagonal_(0)

    while len(selected) < K:
        best_increment = -float("inf")
        best_node = None
        for i in range(sym_matrix.shape[0]):
            if i in selected:
                continue

            inc = 0.0
            for j in selected:
                if i < j:
                    inc += sym_matrix[i, j].item()
                else:
                    inc += sym_matrix[j, i].item()

            if inc > best_increment:
                best_increment = inc
                best_node = i

        selected.add(best_node)

    true_select = [indices_mapping[i] for i in selected]
    return true_select, S_k
```

Integer Quadratic Programming frame selection:

```python
def ILP_select_nodes(matrix, K, shrink_K, rank):
    device = matrix.device
    n = matrix.shape[0]
    sym_matrix = (matrix + matrix.T).to(device)

    if n <= 4 * K:
        indices_mapping = {i: i for i in range(n)}
    else:
        if n > 7200:
            S_k = sym_matrix
        else:
            U, Sigma, Vt = torch.svd(sym_matrix)
            Sigma_k = torch.diag(Sigma[: n // 4])
            U_k = U[:, : n // 4]
            Vt_k = Vt[:, : n // 4]
            S_k = U_k @ Sigma_k @ Vt_k.T

        sparse_matrix, _, indices_mapping = uniform_select_matrix(S_k, M=shrink_K)
        sym_matrix = sparse_matrix

    sym_matrix.fill_diagonal_(0)
    solution, score = cplex_frame_selection(sym_matrix, K, rank)
    true_select = [indices_mapping[i] for i in solution]
    return sorted(true_select)
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
