## Wavelet Packing for Self-Supervised Monocular Depth Estimation

Official PyTorch implementation of the _WavPackNet_ paper: **Wavelet Packing for Self-Supervised Monocular Depth Estimation (ICIP 2025)**,
*Ayoub Rhim, Lei Qin, Rachid Benmokhtar and Xavier Perrotton* from anSWer, Valeo Brain.

Our main contributions: **WavPacking**, **WavUnPacking** blocks and **WavPackNet** are implemented in [sfm_learner/networks/layers/wavpacking.py](./sfm_learner/networks/layers/wavpacking.py) and [sfm_learner/networks/depth/WavPackNet.py](./sfm_learner/networks/depth/WavPackNet.py) respectively. The self-supervised SfM learning framework is adopted from [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm).

## Install

TODO: add requirements.txt


## Datasets

### Cityscapes 
Please follow the instructions from [SfMLearner](https://github.com/tinghuiz/SfMLearner) to prepare the Cityscapes dataset for training.

### KITTI
Please follow the instructions from [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm) to prepare the KITTI dataset for training, validation and evaluation.

## Training

To train WavPackNet from scratch on Cityscape dataset using 192x640 resolution:

```bash
python3 scripts/train.py configs/train_cityscapes_wavpack_MR.yaml
```

To train WavPackNet from scratch on KITTI dataset using 192x640 resolution:

```bash
python3 scripts/train.py configs/train_kitti_wavpack_MR.yaml
```

With a model trained on the Cityscapes dataset, to finetune it on the KITTI dataset:
```bash
python3 scripts/train.py pretrained_cityscapes_model.ckpt
```

## Inference
You can directly run inference on a single image or a directory of images:

```bash
python3 scripts/infer.py --checkpoint <checkpoint.ckpt> --input <image or folder> --output <image or folder> [--image_shape <input shape (h,w)>]
```

## Evaluation

To evaluate a trained model (e.g. our [released models](#models)), provide the `.ckpt` checkpoint, followed optionally by a `.yaml` config file that overrides the configuration stored in the checkpoint, to [scripts/eval.py](./scripts/eval.py).

```bash
python3 scripts/eval.py --checkpoint <checkpoint.ckpt> [--config <config.yaml>]
```

## Models

### Evaluation on 697 KITTI test images with original ground truth

| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | d < 1.25 |
| :--- | :---: | :---: | :---: |  :---: |  :---: |
| [CS_128x416_M](https://huggingface.co/lqin/WavPackNet/resolve/main/CS_128x416_M.ckpt) | 0.116 | 0.811 | 4.902 | 0.198 | 0.865 |
| [CS_192x640_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS_192x640_M.ckpt) | 0.184 | 1.389 | 5.792 | 0.254 | 0.744 |
| [CS_384x1280_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS_384x1280_M.ckpt) |0.187 | 1.493 | 5.891 | 0.253 | 0.737 |
| [K_192x640_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_192x640_M.ckpt) | 0.109 | 0.778 | 4.527 | 0.185 | 0.886 |
| [K_192x640_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_192x640_M%2Bv.ckpt) | 0.110 | 0.840 | 4.762 | 0.198 | 0.868 |
| [K_384x1280_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_384x1280_M.ckpt) | 0.105 | 0.748 | 4.390 | 0.182 | 0.894 |
| [K_384x1280_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_384x1280_M%2Bv.ckpt) | 0.106 | 0.828 | 4.582 | 0.192 | 0.878|
| [CS+K_192x640_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_192x640_M.ckpt) | 0.108 | 0.762 | 4.515 | 0.184 | 0.886|
| [CS+K_192x640_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_192x640_M%2Bv.ckpt) | 0.107 | 0.811 | 4.566 | 0.190 | 0.879|
| [CS+K_384x1280_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_384x1280_M.ckpt) | 0.105 | 0.736 | 4.332 | 0.180 | 0.891|
| [CS+K_384x1280_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_384x1280_M%2Bv.ckpt) | 0.102 | 0.786 | 4.473 | 0.188 | 0.885|

### Evaluation on 652 KITTI test images with improved ground truth

| Model | Abs.Rel. | Sqr.Rel | RMSE | RMSElog | d < 1.25 |
| :--- | :---: | :---: | :---: |  :---: |  :---: |
| [CS_128x416_M](https://huggingface.co/lqin/WavPackNet/resolve/main/CS_128x416_M.ckpt) | 0.145 | 1.009 | 5.018 | 0.195 | 0.815 |
| [CS_192x640_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS_192x640_M.ckpt) | 0.148 | 0.957 | 4.907 | 0.200 | 0.809 |
| [CS_384x1280_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS_384x1280_M.ckpt) | 0.152 | 1.036 | 5.045 | 0.201 | 0.801 |
| [K_192x640_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_192x640_M.ckpt) | 0.076 | 0.402 | 3.428 | 0.117 | 0.936 |
| [K_192x640_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_192x640_M%2Bv.ckpt) | 0.084 | 0.441 | 3.629 | 0.128 | 0.918 |
| [K_384x1280_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_384x1280_M.ckpt) | 0.072 | 0.362 | 3.198 | 0.110 | 0.943 |
| [K_384x1280_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/K_384x1280_M%2Bv.ckpt) | 0.080 | 0.437 | 3.448 | 0.122 | 0.927 |
| [CS+K_192x640_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_192x640_M.ckpt) | 0.076 | 0.406 | 3.443 | 0.116 | 0.936 |
| [CS+K_192x640_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_192x640_M%2Bv.ckpt) | 0.081 | 0.428 | 3.441 | 0.122 | 0.929 |
| [CS+K_384x1280_M.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_384x1280_M.ckpt) | 0.072 | 0.355 | 3.165 | 0.110 | 0.943 |
| [CS+K_384x1280_M+v.ckpt](https://huggingface.co/lqin/WavPackNet/resolve/main/CS%2BK_384x1280_M%2Bv.ckpt) | 0.076 | 0.395 | 3.285 | 0.116 | 0.935 |

## License

The source code is released under the [MIT license](LICENSE).

## References

If you find this code useful, please cite our work:

```
@inproceedings{wavpacknet,
  author = {Rhim, Ayoub and Qin, Lei and Benmokhtar, Rachid and Perrotton, Xavier},
  title = {Wavelet Packing for Self-Supervised Monocular Depth Estimation},
  booktitle = {IEEE Conference on Image Processing (ICIP)},
  primaryClass = {cs.CV}
  year = {2025},
}
```

## Acknowledgement
We thank the the authors and contributors of [**Pytorch Wavelets**](https://github.com/fbcotter/pytorch_wavelets) and [**PackNet-SfM**](https://github.com/TRI-ML/packnet-sfm) for sharing their work.
