## Official Implementations "One-Way Ticket : Time-Independent Unified Encoder for Distilling Text-to-Image Diffusion Models" (CVPR'25)

<img src="assets/teaser.jpg" width="1000px"/>

---

<img src="assets/feature_similarity.jpg" width="1000px"/>

Above a certain threshold of steps, such as 15 steps in SD2.1, the model maintains image generation quality (Fig.a-b) while the features show high similarity (Fig.c). Below this threshold, feature similarity deteriorates along with worse generation quality, accompanied by a degradation in image generation quality as sampling steps reduce. Furthermore, the encoder features consistently exhibit higher similarity than the decoder across all sampling steps (Fig.c). Additionally, the decoder feature always shows much higher variations than encoder (Fig.c).


Hence we use a novel design with 1-step encoder and a 4-step decoder, achieving near 1-step inference. Since the 4-step decoder captures richer semantics, ours aligns the generation quality with multi-step DMs.

# TODO
- [ ] Release inference code and weight
- [ ] Release training code