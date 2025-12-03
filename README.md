<p align="center">
  <h1 align="center">Emergent Outlier View Rejection in Visual Geometry Grounded Transformers</h1>
  <p align="center">
    <a href="https://onground-korea.github.io/">Jisang Han</a><sup>1,2*</sup>
    ·
    <a href="https://sunghwanhong.github.io/">Sunghwan Hong</a><sup>3*</sup>
    ·
    <a href="https://crepejung00.github.io/">Jaewoo Jung</a><sup>1</sup>
    ·
    <a href="https://scholar.google.com/citations?hl=ko&user=7cyLEQ0AAAAJ">Wooseok Jang</a><sup>1</sup>
    ·
    <a href="https://hg010303.github.io/">Honggyu An</a><sup>1</sup>
    ·
    <a href="https://qianqianwang68.github.io/">Qianqian Wang</a><sup>4</sup>
    ·
    <a href="https://scholar.google.com/citations?user=cIK1hS8AAAAJ">Seungryong Kim</a><sup>1†</sup>
    ·
    <a href="https://scholar.google.com/citations?user=YeG8ZM0AAAAJ">Chen Feng</a><sup>2†</sup>
  </p>
  <h4 align="center"><sup>1</sup>KAIST AI, <sup>2</sup>New York University, <sup>3</sup>ETH AI Center, ETH Zurich, <sup>4</sup>UC Berkeley</h4>
  <!-- <p align="center"><sup>‡</sup>Work done during a visiting researcher at New York University&emsp;<sup>*</sup>Equal contributions&emsp;<sup>†</sup>Co-corresponding</p> -->
  <h3 align="center">
    <a href="#">arXiv</a> | 
    <a href="https://github.com/cvlab-kaist/RobustVGGT/releases/download/paper/Emergent.Outlier.View.Rejection.in.Visual.Geometry.Grounded.Transformers.pdf">Paper (High quality)</a> | 
    <a href="https://cvlab-kaist.github.io/RobustVGGT">Project Page</a>
  </h3>
</p>

<p align="center">
  <a href="">
    <img src="assets/teaser.png" alt="Logo" width="80%">
  </a>
</p>

> We reveal that Visual Geometry Grounded Transformers (VGGT) has a built-in ability to detect outliers, which we leverage to perform outlier-view rejection without any fine-tuning.

**What to expect:**

- [x] Demo inference code
- [ ] Evaluation code
- [ ] Visualization code

## Installation

Our code is developed based on pytorch 2.5.1, CUDA 12.1 and python 3.10. 

We recommend using [conda](https://docs.anaconda.com/miniconda/) for installation:

```bash
conda create -n robust_vggt python=3.10
conda activate robust_vggt

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Running Demo

To run the robust reconstruction demo with outlier rejection:

```bash
python robust_vggt.py --image-dir examples/trevi
```

```bash
python robust_vggt.py --image-dir examples/notredame --rej-thresh 0.3
```

## Citation

```
@article{han2024emergent,
  title={Emergent Outlier View Rejection in Visual Geometry Grounded Transformers},
  author={Han, Jisang and Hong, Sunghwan and Jung, Jaewoo and Jang, Wooseok and An, Honggyu and Wang, Qianqian and Kim, Seungryong and Feng, Chen},
  journal={arXiv preprint arXiv:2410.xxxxx},
  year={2024}
}
```

## Acknowledgement
We thank the authors of [VGGT](https://github.com/facebookresearch/vggt) for their excellent work and code, which served as the foundation for this project.
