# Standalone YOLO‑Stutter Inference

This folder is self‑contained: `min_inference.py` only depends on files in this directory.

## 1) Activate the environment

```
source /home/olabode/miniconda3/etc/profile.d/conda.sh
conda activate yolostutter
```

## 2) Put the checkpoints + config in this folder

### Option A — copy from the main repo (fastest)
From the repo root:

```
cp /home/olabode/workplace/YOLO-Stutter/yolo-stutter/saved_models/pretrained_ljs.pth \
   /home/olabode/workplace/YOLO-Stutter/standalone_inference/

cp /home/olabode/workplace/YOLO-Stutter/yolo-stutter/saved_models/decoder_tts_joint \
   /home/olabode/workplace/YOLO-Stutter/standalone_inference/

cp /home/olabode/workplace/YOLO-Stutter/yolo-stutter/utils/vits/configs/ljs_base.json \
   /home/olabode/workplace/YOLO-Stutter/standalone_inference/
```

### Option B — download directly into this folder
If you **don’t** already have the checkpoints, download them here:

```
python -m pip install -q gdown

python - <<'PY'
from pathlib import Path
import shutil
import gdown

out_dir = Path('/home/olabode/workplace/YOLO-Stutter/standalone_inference')

vits_url = 'https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2'
ckpt_url = 'https://drive.google.com/drive/folders/1-iD0D3A5IKPrKGfvIr3age8tGVL_yKnL?usp=sharing'

# download folders
vits_dir = out_dir / 'tmp_vits'
ckpt_dir = out_dir / 'tmp_ckpt'
vits_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

gdown.download_folder(vits_url, output=str(vits_dir), quiet=False, use_cookies=False)
gdown.download_folder(ckpt_url, output=str(ckpt_dir), quiet=False, use_cookies=False)

# move required files into standalone_inference
pretrained = next(vits_dir.rglob('pretrained_ljs.pth'))
decoder = next(ckpt_dir.rglob('decoder_tts_joint'))

shutil.copy2(pretrained, out_dir / 'pretrained_ljs.pth')
shutil.copy2(decoder, out_dir / 'decoder_tts_joint')

print('Downloaded to:', out_dir)
PY
```

You also need the config file:

```
cp /home/olabode/workplace/YOLO-Stutter/yolo-stutter/utils/vits/configs/ljs_base.json \
   /home/olabode/workplace/YOLO-Stutter/standalone_inference/
```

## 3) Run inference

```
python /home/olabode/workplace/YOLO-Stutter/standalone_inference/min_inference.py \
  --audio /home/olabode/workplace/YOLO-Stutter/inference/raw.wav \
  --transcript /home/olabode/workplace/YOLO-Stutter/inference/raw_transcript.txt \
  --output-json /home/olabode/workplace/YOLO-Stutter/inference/output.json
```

### Optional flags
- `--device cuda` for GPU
- `--top-k 10` to return more predictions per chunk

## Expected files in this folder
- `min_inference.py`
- `util.py`
- `pretrained_ljs.pth`
- `decoder_tts_joint`
- `ljs_base.json`
