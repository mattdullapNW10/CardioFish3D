# Trained Models

Store trained model weights and checkpoints here.

## Naming Convention

Use descriptive names with version numbers:
- `model_v1_baseline.pt` — PyTorch model
- `model_v2_segmentation.h5` — TensorFlow/Keras model
- `checkpoint_epoch_50.pt` — Training checkpoint

## Structure

```
models/
├── model_v1.pt
├── model_v2.h5
└── config_v1.json        # Model configuration/hyperparameters
```

## Note

Large model files (>100MB) should be tracked with Git LFS or stored separately.
