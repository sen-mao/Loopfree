## Official Implementations "One-Way Ticket : Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Models" (CVPR'25)

<img src="assets/teaser.jpg" width="1000px"/>

---

<img src="assets/feature_similarity.jpg" width="1000px"/>

Above a certain threshold of steps, such as 15 steps in SD2.1, the model maintains image generation quality (Fig.a-b) while the features show high similarity (Fig.c). Below this threshold, feature similarity deteriorates along with worse generation quality, accompanied by a degradation in image generation quality as sampling steps reduce. Furthermore, the encoder features consistently exhibit higher similarity than the decoder across all sampling steps (Fig.c). Additionally, the decoder feature always shows much higher variations than encoder (Fig.c).

Hence we use a novel design with 1-step encoder and a 4-step decoder (Time-independent Unified Encoder architecture), achieving near 1-step inference. Since the 4-step decoder captures richer semantics, ours aligns the generation quality with multi-step DMs.

---

<img src="assets/loopfree.jpg" width="1000px"/>
Building on our Time-independent Unified Encoder (TiUE) architecture, we introduce a loop-free distillation approach. 

### Update
- **2025.04.22**: Release the pre-trained models for [Loopfree SD1.5](https://huggingface.co/senmaonk/loopfree-sd1.5) and [Loopfree SD2.1-Base](https://huggingface.co/senmaonk/loopfree-sd2.1-base), and inference code. 😀
- **2025.03.15**: This repo is created.

### TODO
- [x] Release inference code and weight
- [ ] Release training code

---

### Dependencies and Installation

```
# git clone this repository
git clone https://github.com/sen-mao/Loopfree.git
cd Loopfree

# create new anaconda env
conda create -n loopfree python=3.8 -y
conda activate loopfree

# install python dependencies
pip3 install -r requirements.txt
```

### Inference

```
# Loopfree SD1.5
python loopfree.py --pretrained_model_name_or_path senmaonk/loopfree-sd1.5 \
                   --output_dir loopfree-sd1.5 --use_parallel
```

```
# Loopfree SD2.1-Base
python loopfree.py --pretrained_model_name_or_path senmaonk/loopfree-sd2.1-base \
                   --output_dir loopfree-sd2.1-base --use_parallel
```