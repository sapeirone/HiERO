# Zero-Shot HiERO evalution on Ego4d Goal-Step

## Step-Grounding

**Step grounding using EgoVLP features**:
```sh
python -m ego4d_goalstep.eval_grounding --features egovlp
```

**Step grounding using HiERO (Omnivore)**:
```sh
python -m ego4d_goalstep.eval_grounding --features omnivore_video_swinl --ckpt checkpoints/hiero_omnivore.pth
```

**Step grounding using HiERO (EgoVLP)**:
```sh
python -m ego4d_goalstep.eval_grounding --features egovlp --ckpt checkpoints/hiero_egovlp.pth
```

### Results

| Config           | Weights | R1 @ IoU = 0.30 | R5 @ IoU = 0.30 | R1 @ IoU = 0.50 | R5 @ IoU = 0.50 |
|------------------|---------|-----------------|-----------------|-----------------|-----------------|
| EgoVLP               | n/a     | 12.65 | 29.93 | 8.85	| 20.28 |
| **HiERO (Omnivore)** | [hiero_omnivore.pth](https://www.sapeirone.it/data/hiero/ckpt/hiero_omnivore.pth) | 10.40 | 27.36 | 7.06 | 18.24 |
| **HiERO (EgoVLP)**   | [hiero_egovlp.pth](https://www.sapeirone.it/data/hiero/ckpt/hiero_egovlp.pth)     | 13.83 | 33.45 | 9.72 | 23.11 |

> [!WARNING]
> **NOTE:** Results reported in the previous table are slightly higher compared to those reported in the paper due to a bug in the handling of grouped videos that was fixed during code cleanup before the release.