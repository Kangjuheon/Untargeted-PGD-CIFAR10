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

```

## Notes
- PGD is a stronger iterative variant of FGSM.
- Attack parameters: epsilon (eps), step size (alpha), and iterations (iters).
