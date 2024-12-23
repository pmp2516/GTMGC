# About this Fork
This is a fork of [GTMGC](https://github.com/Rich-XGK/GTMGC) for the purposes of academic experimentation and tinkering.

Please see the original repository for a stable, working implementation of the work in the paper.

# Training and Evaluating
The functionality of GTMGC's PyTorch models has been slightly extended for implementation of my experiments, which are not documented in detail here. **DISCLAIMER: These modifications do not constitute an improvement over GTMGC, and tend to perform worse on most metrics.**

To train a model with the revised architecture, set the value of `num_revised_layers` in the configuration to the desired depth of the network. The loss function may need to be adjusted to suit your needs. To train a model with the weight-sharing architecture, set `share_weights=True` in the configuration (this is what is provided as of the last commit).

To evaluate such a model, use the evaluation method `"GTMGC_Revised"`. In practice this differs only slightly from the original evaluation setup.

## Citation
If you intend to use GTMGC in your research, please use the original reposirory, https://github.com/Rich-XGK/GTMGC, and cite the original paper.

```bibtex
@inproceedings{xu2024gtmgc,
  title={GTMGC: Using Graph Transformer to Predict Molecule’s Ground-State Conformation},
  author={Xu, Guikun and Jiang, Yongquan and Lei, PengChuan and Yang, Yan and Chen, Jim},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

