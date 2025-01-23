<div align="center">
<h1>MD-Dose </h1>
This is the codebase for [MD-Dose: A diffusion model based on the Mamba for radiation dose prediction](https://ieeexplore.ieee.org/abstract/document/10822581).
</div>

## Envs. for Pretraining
- Python 3.10.13

  - `conda create -n your_env_name python=3.10.13`

- torch 2.1.1 + cu118
  - `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

- Requirements: vim_requirements.txt
  - `pip install -r vim/vim_requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal_conv1d`
  - `pip install -e mamba`
## Train Your MD-Dose

`python train.py(Training without a structural encoder)`

`python train_struc.py(Training with a structural encoder)`

## Acknowledgement :heart:
This project is based on Vim ([paper](https://arxiv.org/abs/2401.09417), [code](https://github.com/hustvl/Vim)), Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)). Thanks for their wonderful works.

## Citation
If you find MD-Dose is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@inproceedings{fu2024md,
  title={MD-Dose: A diffusion model based on the Mamba for radiation dose prediction},
  author={Fu, Linjie and Li, Xia and Cai, Xiuding and Wang, Xueyao and Shen, Yali and Yao, Yu},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={911--918},
  year={2024},
  organization={IEEE}
}
```
