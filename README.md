# Adversarial Attacks on CIFAR Datasets

This project demonstrates how to train a model with the LGN-AR regularizer and perform FGSM and PGD attacks, along with the computation of LPIPS and NIQE scores.

This project uses the `advertorch` and `Foolbox` libraries.

## Attacks

- **FGSM Attack**: Implemented in `fgsm_attack.py`.
- **PGD Attack**: Implemented in `pgd_attack.py`.

## Dependencies

- PyTorch
- torchvision
- advertorch
- matplotlib
- lpips 
- tqdm 
- scikit-image

## Directory Structure

- **attacks/**: Contains scripts for FGSM and PGD attacks.
- **models/**: Contains model definitions.
- **metrics/**: 
  - `compute_lpips.py`: Script for computing LPIPS scores between robust and standard model gradients.
  - `visualize_lpips.py`: Script for visualizing LPIPS scores computed in `compute_lpips.py`.
  - `compute_niqe.py`: Script for computing NIQE scores.
  - `visualize_niqe.py`: Script for visualizing NIQE results.
- **utils/**: Contains utility functions.

## Usage

1. **Install the dependencies**:
   ```bash
   pip install torch torchvision advertorch matplotlib lpips tqdm scikit-image


## How to Run
1. Train the model:
   python train_cifar10.py

2. Run the desired attack script:
   python attacks/fgsm_attack.py
   python attacks/pgd_attack.py

3. Compute LPIPS Scores: Run the script to compute LPIPS scores:
   python metrics/compute_lpips.py

4. Visualize LPIPS Results: After computing the LPIPS scores, run the script to visualize the results:
   python metrics/visualize_lpips.py

5. Compute NIQE Scores: Run the script to compute NIQE scores:
   python metrics/compute_niqe.py

6. Visualize NIQE Results: After computing the NIQE scores, run the script to visualize the results:
   python metrics/visualize_niqe.py

## Related Research

1. @inproceedings{srinivas2023pags,
  author    = {Suraj Srinivas and Sebastian Bordt and Himabindu Lakkaraju},
  title     = {Which Models have Perceptually-Aligned Gradients? An Explanation via Off-Manifold Robustness},
  booktitle = {NeurIPS},
  year      = {2023}
}
2.Olivier J H´enaff, Robbe LT Goris, and Eero P Simoncelli. Perceptual straightening of natural videos. Nature neuroscience, 22(6):984–991, 2019
3. We use the LPIPS metric from Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR'18.

4. We use the diffusion models for both cifar10 and cifar100 from Karas et al. "Elucidating the Design Space of Diffusion-Based Generative Models (EDM)", NeurIPS'22. 
