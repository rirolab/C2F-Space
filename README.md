<div align="center">
    <h1> <a>C2F-Space: Coarse-to-Fine Space Grounding for Spatial Instructions using Vision-Language Models</a></h1>
</div>
<p align="center">
  <a href="https://arxiv.org/abs/2511.15333">Paper</a> •
  <a href="https://github.com/rirolab/C2F-Space">Code</a> •
  <a href="#bibtex">BibTex</a>
</p>


C2F-Space is a coarse-to-fine space-grounding framework that performs coarse reasoning via propose-validate VLM prompting and refines predictions through superpixel-wise residual learning for precise local geometric reasoning.
<p align="center">
    <img width="500" alt="image" src="./assets/images/image.png">
</p>
</details>

## Repository Structure
```
C2F-Space
├── coarse_vlm: VLM based space reasoning
├── fine_refinement: Superpixel-level space refinement
├── scene_generation: Pybullet simulation
└── sgg: Grounded-Segment-Anything based object indentification

```


## Installation
Clone this repository:
```bash
git clone --recurse-submodules https://github.com/rirolab/C2F-Space.git
```

Create a conda environment:
```bash
conda create -n c2f_space python=3.8
conda activate c2f_space
```

Install dependencies:
```bash
bash install.sh
```

Get weights for Grounded-Segment-Anything:
```bash
cd sgg/Grounded-Segment-Anything/segment_anything
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Running the demo code
First, you need to set "OPENAI_API_KEY" to use the semantic parser:
```bash
export OPENAI_API_KEY='YOUR API KEY'
```
Please, refer [OpenAI](https://openai.com/) to get an API key.

Then, execute
```bash
python demo.py --ckpt_folder ...
```
You need to pass `ckpt_folder` to indicate the trained refinement model. 
You can check the result of grounded object pose in `result.png`.
If you want to record videos, please add `--record`.


## Troubleshooting
If you are having trouble installing opencv, try the following:
```
sudo apt-get install -y libglib2.0-0
sudo apt-get install -y libsm6
```

## Acknowledgement
We thank to open source repositories: [GraphGPS](https://github.com/rampasek/GraphGPS), [cliport](https://github.com/cliport/cliport), and [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything).

## BibTex
```
@article{oh2025c2f,
  title         = {C2F-Space: Coarse-to-Fine Space Grounding for Spatial Instructions using Vision-Language Models},
  author        = {Oh, Nayoung and Kim, Dohyun and Bang, Junhyeong and Paul, Rohan and Park, Daehyung},
  journal       = {arXiv preprint arXiv:2511.15333},
  year          = {2025},
  url           = {https://arxiv.org/abs/2511.15333}
}
``` 