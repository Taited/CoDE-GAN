## CoDE-GAN: Content Decoupled and Enhanced GAN for Sketch-guided Flexible Fashion Editing


https://github.com/user-attachments/assets/8692c9ec-a53d-4fe9-95d6-9f5297b51867

<a href="https://taited.github.io/codegan-project" target="_blank">
  <img src="https://img.shields.io/badge/Project-Page-Green">
</a>
<a href="https://dl.acm.org/doi/10.1145/3712063" target="_blank">
  <img src="https://img.shields.io/badge/Paper-Published-red">
</a>

This is the official implementation of CoDE-GAN (ToMM 2025). The trained weights and a more detailed README are in progress.

## Environment Requirements
We have tested the requirements on ``torch==2.4.1`` in ``cuda-11.8``. But we believe this repository should have no speciall requirements for torch or cuda versions.
1. Create a conda environment and install a torch that compatible with your cuda.
```bash
conda create -n codegan python=3.10
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```
2. Install mmcv for dependencies.
```bash
pip install openmim
mim install mmcv-full
```
3. Install other requirements.
```bash
pip install -r requirements.txt
```

## Inference
```bash
python inference-img.py
```
