## Requirements
```bash
apt install ffmpeg
conda create -n imojen python==3.9
conda activate imojen
pip install torch torchvision torchaudio
pip install --upgrade "pip<24.1"
pip install -r requirements.txt
```

## Download Checkpoints
```bash
./tools/dlmodels.sh
```
## Training
```bash
(leave all training samples in a folder - e.g. Imogen)
sh run.sh Imogen
```

## Inference
```bash
python inference.py --vocal Imogen --pitch jen
(for the --pitch, there are three options, crepe / rmvpe / jen)
```

