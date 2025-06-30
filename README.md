### Installation

```bash
pip install -e .
pip install flash-attn --no-build-isolation
cd transformers
pip install -e .
```

if you run into flash attention error, try to downgrade torch to 2.5.1 and torchvision to 0.20.1:

```bash
pip uninstall torch torchvision 
pip install torch==2.5.1 torchvision==0.20.1 
```
### Download models

download original anole model
```bash
huggingface-cli download --resume-download GAIR/Anole-7b --local-dir Anole-7b --local-dir-use-symlinks False
```

downlaod our checkpoints
```bash
git clone https://huggingface.co/vlm-reasoning-cot/anole-7b-hf-2025-trained-ckpt-16500
git lfs pull
```

### Register Model

Go to `config.py` find supported_VLM (line 1426) and change the model_path to the path of the model.


### Run Eval

```bash
torchrun --nproc-per-node=8 run.py --data MMVP --model anole_15000
```

Datasets:

```
MathVision_MINI MathVerse_MINI MathVista_MINI EMMA_COT MMVP MMMU_DEV_VAL VisuLogic BLINK
```
