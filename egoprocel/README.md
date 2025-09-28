# HiERO evalution on EgoProceL

HiERO can address the Procedure Learning task on EgoProceL in zero-shot.

## 🔎 Getting started

Download the official EgoProceL [annotations](https://drive.google.com/drive/folders/17u7ReqPOJ29lbVZT8PDEd5dN17VBqS5Q), videos and our pre-extracted [Omnivore]() and [EgoVLP]() features.

**Baseline evaluation (using omnivore or egovlp):**
```sh
python -m egoprocel.baseline_eval --features egovlp
```

**HiERO evaluation (using omnivore or egovlp features)**:
```sh
python -m egoprocel.hiero_eval --features egovlp --ckpt path/to/model.pth
```

## 🐘 Model and features zoo

HiERO is built on existing features extractors, i.e., Omnivore and EgoVLP. 
For evaluation on EgoProceL, we provide **pre-extracted features** using these backbones and their officially released weights, as well as the **HiERO checkpoints**.

| Model | Checkpoint | Features | F1 | IoU |
|-------|------------|----------|----|-----|
| Omnivore | - | [url](https://www.sapeirone.it/data/hiero/egoprocel_features/egoprocel_omnivore.zip) | 39.1 | 22.0 |
| EgoVLP | - | [url](https://www.sapeirone.it/data/hiero/egoprocel_features/egoprocel_egovlp.zip) | 40.0 | 21.9 |
| HiERO (Omnivore) | [hiero_omnivore.pth](https://www.sapeirone.it/data/hiero/ckpt/hiero_omnivore.pth) | [url](https://www.sapeirone.it/data/hiero/egoprocel_features/egoprocel_omnivore.zip) | 44.0 | 24.5 |
| HiERO (EgoVLP) | [hiero_egovlp.pth](https://www.sapeirone.it/data/hiero/ckpt/hiero_egovlp.pth) | [url](https://www.sapeirone.it/data/hiero/egoprocel_features/egoprocel_egovlp.zip) | 44.5 | 25.3 |


## 🙏 Acknowledgements

We thank the [EgoProceL](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning) authors for releasing the dataset.
Portions of the code used for EgoProceL evaluation were taken from the [EgoProceL](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning) repository and from the [OPEL](https://openreview.net/forum?id=leqD3bJ4Ly) code release.