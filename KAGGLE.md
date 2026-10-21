# Running this on Kaggle (and Colab)

## Can you run shell commands?

Yes, but there is no terminal — Kaggle gives you notebook cells only. Shell commands run with `!`
or a `%%bash` cell, which is all these scripts need:

```python
!python train.py --freeze-encoder --max-steps 200 --subset 500
```

## What persists, and what does not

**Treat `/kaggle/working` as ephemeral.** It holds your files during a session and is kept as the
notebook's output *only if the version is saved*; a fresh session starts with it empty, and a
session killed at the 12-hour cap or by an idle timeout takes anything unsaved with it. To reuse a
previous run's files from `/kaggle/working` you must attach that notebook version's output as an
input data source — which is fiddly and easy to forget mid-run.

So the durable store is the **Hugging Face Hub**, which is why `--hub-repo` exists:

| event | local `ckpt/last.pt` | Hub `last.pt` |
|---|---|---|
| 12-hour session cap | gone unless saved | **safe** |
| idle timeout / browser closed | gone unless saved | **safe** |
| 30 GPU-hours exhausted | gone unless saved | **safe** |
| moving Kaggle → Colab | not visible | **safe** |

Always pass both flags on a real run, and the run becomes interruption-proof:

```python
!python train.py --unfreeze-top 8 --lr 2e-5 \
    --out /kaggle/working/ckpt \
    --hub-repo YOURNAME/sinhala-gemma-ctc \
    --resume ckpt/last.pt
```

`--save-every 500` pushes to the Hub on each save; `--resume` pulls from the Hub when the local
file is missing. Re-running that identical command after any interruption continues the run.

Lower `--save-every` to 200 if you are near the end of your weekly quota — an unpushed hour is the
only thing you can actually lose.

## Use "Save & Run All", not an interactive session, for long runs

An interactive session can stop when you close the browser. **Save Version → Save & Run All
(Commit)** runs the notebook headless for up to 12 hours and keeps the outputs. You don't see
intermediate logs until it finishes, which is fine once stage 1 has proved the pipeline. Use
interactive sessions for `probe.py`, `vocab.py` and the smoke test; commit the long runs.

## What to upload

The five `.py` files plus `requirements.txt`. Two ways:

**A — Kaggle Dataset (no GitHub needed).** Zip this folder, then *Datasets → New Dataset → Upload*.
Name it `sinhala-asr-code`. It mounts read-only at `/kaggle/input/sinhala-asr-code/`, so copy it
out before running, because the scripts write `arch.json`, `vocab.json` and checkpoints:

```python
!cp /kaggle/input/sinhala-asr-code/*.py /kaggle/input/sinhala-asr-code/requirements.txt /kaggle/working/
%cd /kaggle/working
```

Re-uploading a new dataset version is the annoying part; prefer B if you expect to edit code.

**B — a private GitHub repo.** Easier to iterate:

```python
!git clone https://github.com/YOU/sinhala_asr.git /kaggle/working/code
%cd /kaggle/working/code
```

Do **not** upload the dataset itself. `stratified_v7` streams from the Hub; 17.2 GB is not worth a
session's disk or your upload bandwidth.

## Session setup, in order

1. **Notebook settings → Accelerator → GPU P100** (one pooled ~30 h/week; T4 ×2 buys no extra
   quota, and this script is single-GPU).
2. **Notebook settings → Internet → On.** Required for `pip`, for streaming the dataset, and for
   the Hub push. Needs phone verification on the account.
3. **Add-ons → Secrets → Add secret**, name `HF_TOKEN`, value a Hub token with **write** access.

## Two notebooks, not one

A Kaggle **CPU session does not spend GPU quota**, and `probe.py` / `vocab.py` need no GPU. Running
them in your training notebook burns ~20 minutes of a 30-hour weekly budget for nothing.

| | notebook A — "asr-setup" | notebook B — "asr-train" |
|---|---|---|
| Accelerator | **None (CPU)** | **GPU P100** |
| Internet | On | On |
| `HF_TOKEN` attached | yes | yes |
| runs | `probe.py`, `vocab.py` | `train.py`, `evaluate.py` |
| Save Version | not needed | **yes, for the long runs** |

The two share results through the Hub repo, the same way checkpoints do.

### Notebook A — setup (CPU, run once, ~30 min)

```python
# A1 - code + deps
!git clone https://github.com/YOU/sinhala_asr.git /kaggle/working/code
%cd /kaggle/working/code
!pip install -q -r requirements.txt
# Kaggle's preinstalled transformers is too old for embedding_gemma2 (added in 5.19.0).
# Confirm the upgrade actually took effect:
!python -c "import transformers; print(transformers.__version__)"   # must be >= 5.19
```

**`KeyError: 'embedding_gemma2'`** from `AutoConfig` means that upgrade did not happen. No kernel
restart is needed as long as the scripts run via `!python ...` -- each is a fresh process that picks
up the new version. Only an `import transformers` *inside* a notebook cell needs
*Run -> Restart session* after upgrading.

```python
# A2 - Hub login, and prove the token works before anything long
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, whoami
login(UserSecretsClient().get_secret("HF_TOKEN"))
print(whoami()["name"])
```

```python
# A3 - architecture -> arch.json
!python probe.py
```

```python
# A4 - vocabulary + the frame-rate GATE -> vocab.json
!python vocab.py --report --limit 20000
```

```python
# A5 - park both on the Hub so notebook B (and Colab) can fetch them
from huggingface_hub import upload_file
REPO = "YOURNAME/sinhala-gemma-ctc"
for f in ("arch.json", "vocab.json"):
    upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=REPO)
    print("pushed", f)
```

Read A4's verdict before going further. If the p95 headroom fails, switch to subword units now --
training on character units that cannot fit the frames wastes the whole budget.

Also copy `arch.json` and `vocab.json` into your local repo and commit them; they are a few hundred
bytes and it saves ever running notebook A again.

### Notebook B — training (GPU P100)

```python
# B1 - code + deps
!git clone https://github.com/YOU/sinhala_asr.git /kaggle/working/code
%cd /kaggle/working/code
!pip install -q -r requirements.txt
```

```python
# B2 - Hub login
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))
```

```python
# B3 - fetch what notebook A produced (skip if they are committed in the repo)
from huggingface_hub import hf_hub_download
import shutil
REPO = "YOURNAME/sinhala-gemma-ctc"
for f in ("arch.json", "vocab.json"):
    shutil.copy(hf_hub_download(repo_id=REPO, filename=f), f)
!ls -la arch.json vocab.json
```

```python
# B4 - SMOKE TEST. Read rows/s from this before anything longer.
!python train.py --freeze-encoder --max-steps 200 --subset 500 --out /kaggle/working/ckpt
```

```python
# B5 - epoch time from the measured rows/s
rows_per_sec = 10.0          # <- replace with what B4 printed
print(f"one epoch = {108278/rows_per_sec/3600:.1f} h")
```

```python
# B6 - head-only baseline (~1.2 epochs)
!python train.py --freeze-encoder --max-steps 4000     --out /kaggle/working/ckpt --hub-repo YOURNAME/sinhala-gemma-ctc
```

```python
# B7 - the real run. Re-run this same cell after each 12-hour cap; --resume pulls from the Hub.
!python train.py --unfreeze-top 8 --lr 2e-5 --max-steps 12000     --out /kaggle/working/ckpt --hub-repo YOURNAME/sinhala-gemma-ctc     --resume ckpt/last.pt
```

```python
# B8 - score it
!python evaluate.py --ckpt /kaggle/working/ckpt/last.pt --split test --limit 2000
```

**Run B4 interactively** (you need to see `rows/s`). **Commit B7** via Save Version -> Save & Run
All, because an interactive session can die when you close the browser. Comment out the cells you
are not running in a committed version -- a commit runs every cell top to bottom.

## Colab differences

Same commands. Three changes:

- Mount nothing; `!git clone` into `/content` and `--out /content/ckpt`.
- Replace the Kaggle Secrets cell with `from huggingface_hub import notebook_login; notebook_login()`.
- Colab's free GPU varies. If it hands you something older than a T4, fp16 still works but `rows/s`
  will differ — take the epoch-time measurement on whichever platform will run stage 3.

Because both platforms resume from the same `--hub-repo`, you can start on Colab and finish on
Kaggle with no copying.
