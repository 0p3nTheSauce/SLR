# Models

The models in this repo come from 3 sources:
- [PyTorch vision library](https://github.com/pytorch/vision)
- [SlowFast GitHub repo](https://github.com/facebookresearch/SlowFast)
- [Detectron2 repo](https://github.com/facebookresearch/detectron2)


## PyTorch vision models

No extra installation is needed, the models are native in [PyTorch vision](https://docs.pytorch.org/vision/stable/models.html#video-classification), and weights are downlaoded automatically. 

This includes the following models:
- [S3D](./pytorch_s3d.py)
- [R3D_18](./pytorch_r3d.py)
- [R(2+1)D_18](./pytorch_r3d.py)
- [Swin3D_T](./pytorch_swin3d.py)
- [Swin3D_S](./pytorch_swin3d.py)
- [Swin3D_B](./pytorch_swin3d.py)
- [MViTv2_S](./pytorch_mvit.py)
- [MViTv2_S_e](./pytorch_mvit.py)
- [MViTv1_B](./pytorch_mvit.py)



Citation:
```bibtex
@software{torchvision2016,
    title        = {TorchVision: PyTorch's Computer Vision library},
    author       = {TorchVision maintainers and contributors},
    year         = 2016,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/pytorch/vision}}
}
```

## SlowFast models

[MViTv2_S_16x4](./og_mvit.py) and [MViTv2_B_32x3](./og_mvit.py) come from the [SlowFast repository](https://github.com/facebookresearch/SlowFast/tree/main/projects/mvitv2). These models are subject to their [license](./mvit/SLOWFAST_LICENSE.md).

To download the weights, use:
```bash
cd models
mkdir -p ./mvit/weights/
wget -P ./mvit/weights/ https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth
wget -P ./mvit/weights/ https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_B_32x3_k400_f304025456.pyth
```

Citation:
```bibtex
@inproceedings{li2021improved,
    title={MViTv2: Improved multiscale vision transformers for classification and detection},
    author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
    booktitle={CVPR},
    year={2022}
}

@inproceedings{fan2021multiscale,
    title={Multiscale vision transformers},
    author={Fan, Haoqi and Xiong, Bo and Mangalam, Karttikeya and Li, Yanghao and Yan, Zhicheng and Malik, Jitendra and Feichtenhofer, Christoph},
    booktitle={ICCV},
    year={2021}
}
```

## Detectron2 models

Some experimental models use a [2D pretrained MViT](https://github.com/facebookresearch/detectron2/tree/main/projects/MViTv2). These are:
- [MVirTed_t](./sep_mvit_bert.py)
- [MVirTed_t_MAE](./mvirted_mae.py)

These models are subject to the detectron2 [licence](./mvit/DETECTRON2_LICENSE.md).

To note: the weights from the image classification [repo](https://github.com/facebookresearch/mvit) may be an alternative, but would require some tweaking to the model setup as we used the object detection weights.

Weights for the B, S and T ImageNet 1K instatiations (pretrained on COCO) can be downloaded as follows:

```bash
wget -p ./mvit/weights/ -O ./mvit/weights/MViTV2-T_IN1K.pkl https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_t_3x/f308344828/model_final_c6967a.pkl

wget -p ./mvit/weights/ -O ./mvit/weights/MViTV2-S_IN1K.pkl https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_s_3x/f308344647/model_final_279baf.pkl

wget -p ./mvit/weights/ -O ./mvit/weights/MViTV2-B_IN1K.pkl https://dl.fbaipublicfiles.com/detectron2/MViTv2/cascade_mask_rcnn_mvitv2_b_3x/f308109448/model_final_421a91.pkl
```

Citation:
```bibtex
@inproceedings{li2021improved,
    title={MViTv2: Improved multiscale vision transformers for classification and detection},
    author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
    booktitle={CVPR},
    year={2022}
}
```