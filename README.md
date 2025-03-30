# PGD Untargeted Attack on CIFAR-10 (ConvNeXt)

This project implements an **untargeted PGD (Projected Gradient Descent)** adversarial attack against a fine-tuned **ConvNeXt-Tiny** model trained on CIFAR-10.

## Overview

- Model: ConvNeXt-Tiny pretrained on ImageNet
- Training: Only the final classifier layer trained on CIFAR-10
- Attack: Untargeted PGD with multiple steps of perturbation
- Evaluation: Accuracy on clean and adversarial examples

## Files

- `test.py`: Main script
- `requirements.txt`: Required libraries

## How to Run

```bash
pip install -r requirements.txt
python test.py
```

## Example Output
```bash
[Clean Accuracy] 90.15%
Epoch 1: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:33<00:00, 23.24it/s, loss=0.324]
[Epoch 1] Avg Loss: 0.5755
Epoch 2: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:35<00:00, 22.34it/s, loss=0.283] 
[Epoch 2] Avg Loss: 0.3804
Epoch 3: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:35<00:00, 22.19it/s, loss=0.516] 
[Epoch 3] Avg Loss: 0.3506
Epoch 4: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:33<00:00, 23.00it/s, loss=0.214] 
[Epoch 4] Avg Loss: 0.3328
Epoch 5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 782/782 [00:34<00:00, 22.81it/s, loss=0.385] 
[Epoch 5] Avg Loss: 0.3202
Clean Evaluation: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:11<00:00,  3.63it/s] 

[Clean Accuracy] 90.15%
PGD Attack Evaluation: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:02<00:00,  3.05s/it] 

[PGD Untargeted Attack Accuracy] eps=0.03, alpha=0.007, iters=10 → 0.00%
```

## Notes
- PGD is a stronger iterative variant of FGSM.
- Attack parameters: epsilon (eps), step size (alpha), and iterations (iters).
