# MBA clean-room Docker environment

This bundle was reconstructed from the environment report captured on 2026-06-16.
It targets the exact core runtime that successfully imported and ran the pipeline:

- Ubuntu 22.04 / Python 3.10.13
- PyTorch 2.2.0, torchvision 0.17.0, torchaudio 2.2.0
- CUDA 12.1, cuDNN 8.9
- Transformers 4.45.2 / Tokenizers 0.20.1
- FlashAttention 2.5.9.post1
- BitsAndBytes 0.43.3
- 4 or 8 RTX A6000 GPUs

The original Docker image digest was not included in the report. The Dockerfile therefore
uses the matching official PyTorch base image rather than claiming byte-identical image
reproduction.

## Important findings from the old container

The working process imported most Hugging Face packages from the user site under
`/home/hchoi/.local`, while PyTorch came from `/opt/conda`. The old environment also had
unrelated/conflicting packages (`vllm-flash-attn`, CUDA 13 Python wheels, both CPU and GPU
FAISS, incomplete Mistral/vLLM extras). They were not required by the three pipeline scripts
and are intentionally excluded from this clean image.

The FActScore checkout was not pristine. Commit
`f28272deffcf33efc1f1117d5479c10bb75221a9` had modified `atomic_facts.py`,
`factscorer.py`, and `retrieval.py`, plus custom `llama_lm.py`, `local_lm.py`, and
`/workspace/wiki_fact_scorer.py`. Package pins alone are therefore insufficient: copy the
custom source files from the known-good container or commit them to the project repository.

## 1. Export custom source from the old container

Copy `export_known_good_runtime.sh` into the old container and run:

```bash
bash export_known_good_runtime.sh /workspace/mba_known_good_overlay.tar.gz
```

On the Docker host:

```bash
docker cp <old-container-id>:/workspace/mba_known_good_overlay.tar.gz .
tar -xzf mba_known_good_overlay.tar.gz --strip-components=1 -C /path/to/mba
```

This archive excludes the 70+ GiB wiki databases and model weights. Keep or mount those
separately.

## 2. Required project layout

The host directory mounted at `/workspace` must contain at least:

```text
2_sft.py
3_grpo.py
4_eval.py
wiki_fact_scorer.py
FActScore/factscore/...
FActScore/data/enwiki-20260101-fts.db
mistral-7b-instruct/...
data/mba/expert_data_verified/...
data/ADEChallengeData2016/images/validation/...
data/mba/mbalibrary/config.json
data/mba/mbalibrary/market_evidence.faiss
data/mba/mbalibrary/market_evidence_meta.jsonl
```

Large models and databases should be mounted, not copied into the Docker image.

## 3. Build

Place this reproduction directory in the repository, for example `repro/`, then run:

```bash
cd repro
docker build -t mba:torch2.2-cu121 .
```

FlashAttention is compiled for RTX A6000 compute capability 8.6. Building can take several
minutes.

## 4. Run

From the project root:

```bash
mba_WORKSPACE="$PWD" \
HF_CACHE="$HOME/.cache/huggingface" \
bash repro/run_docker.sh
```

Equivalent manual command:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/workspace/.cache/huggingface" \
  -w /workspace \
  mba:torch2.2-cu121 bash
```

Do not add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the current A6000 platform
reported it as unsupported.

## 5. Verify before training

Inside the new container:

```bash
python /usr/local/bin/verify_mba_environment.py \
  --workspace /workspace \
  --expected-gpus 8 \
  --strict-versions
```

To compare critical source/data hashes with the captured container, first copy
`known-good-critical.sha256` to `/workspace/repro/`, then run:

```bash
python /usr/local/bin/verify_mba_environment.py \
  --workspace /workspace \
  --expected-gpus 8 \
  --strict-versions \
  --check-hashes \
  --hash-manifest /workspace/repro/known-good-critical.sha256
```

A hash mismatch does not automatically mean the newer code is wrong; it means it is not the
same file that was captured in the working environment report.

## 6. Execute the pipeline

SFT, 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 /workspace/2_sft.py
```

GRPO, 8 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u /workspace/3_grpo.py
```

Evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u /workspace/4_eval.py --model_type base

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u /workspace/4_eval.py --model_type sft

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u /workspace/4_eval.py --model_type grpo
```

## 7. Offline or air-gapped hosts

The Docker build needs access to PyPI/GitHub for packages and the spaCy model. Model weights
can be reused by mounting the existing Hugging Face cache. For a fully offline build, first
create a wheelhouse on an online host and replace the `pip install` stages with
`--no-index --find-links=/wheelhouse`.

## Files in this bundle

- `Dockerfile`: clean CUDA/PyTorch runtime
- `requirements-runtime.txt`: direct runtime pins from the working container
- `environment.yml`: non-Docker Conda alternative
- `verify_environment.py`: versions, CUDA, assets, imports, and optional hash verification
- `run_docker.sh`: reproducible Docker run command
- `docker-compose.yml`: Compose alternative
- `export_known_good_runtime.sh`: export custom source from the old container
- `known-good-critical.sha256`: captured critical checksums
- `*-observed.*`: raw forensic records from the uploaded environment report
