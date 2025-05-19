<div align="center">
<img align="left" height="75" style="margin-left: 20px" src="assets/hiero.png" alt="">

# HiERO: understanding the hierarchy of human behavior enhances reasoning on egocentric videos

**🌴 ICCV 2025**

[Simone Alberto Peirone](https://scholar.google.com/citations?user=K0efPssAAAAJ), [Francesca Pistilli](https://scholar.google.com/citations?user=7MJdvzYAAAAJ), [Giuseppe Averta](https://scholar.google.com/citations?user=i4rm0tYAAAAJ)

</div>

<div align="center">
<a href='https://arxiv.org/abs/2505.12911' style="margin: 10px"><img src='https://img.shields.io/badge/Paper-Arxiv:2505.12911-red'></a>
<a href='https://sapeirone.github.io/HiERO/' style="margin: 10px"><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<br>
✨ <i>Code and checkpoints coming soon!</i> ✨
</div>

<br>

<div align="center">
<figure>
<img style="max-height: 480px" src="assets/teaser_animated.gif"/>
<br>
<caption>Zero-Shot procedure step localization with HiERO. Given a long video, HiERO computes segment-level features that encode the functional dependencies between its actions at different scales, enabling the detection of procedure steps through a simple clustering in feature space.</caption>
</figure>
</div>


## Abstract

Human activities are particularly complex and variable, and this makes challenging for deep learning models to reason about them. However, we note that such variability does have an underlying structure, composed of a hierarchy of patterns of related actions. We argue that such structure can emerge naturally from unscripted videos of human activities, and can be leveraged to better reason about their content. We present HiERO, a weakly-supervised method to enrich video segments features with the corresponding hierarchical activity threads. By aligning video clips with their narrated descriptions, HiERO infers contextual, semantic and temporal reasoning with an hierarchical architecture. We prove the potential of our enriched features with multiple video-text alignment benchmarks (EgoMCQ, EgoNLQ) with minimal additional training, and in zero-shot for procedure learning tasks (EgoProceL and Ego4D Goal-Step). Notably, HiERO achieves state-of-the-art performance in all the benchmarks, and for procedure learning tasks it outperforms fully-supervised methods by a large margin (+12.5% F1 on EgoProceL) in zero shot. Our results prove the relevance of using knowledge of the hierarchy of human activities for multiple reasoning tasks in egocentric vision.

## Acknowledgements
This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them. We acknowledge the CINECA award under the ISCRA initiative, for the availability of high performance computing resources and support. Antonio Alliegro and Tatiana Tommasi also acknowledge the EU project ELSA - European Lighthouse on Secure and Safe AI (grant number 101070617).


## Cite Us

```
@article{peirone2025hiero,
  title={HiERO: understanding the hierarchy of human behavior enhances reasoning on egocentric videos},
  author={Peirone, Simone Alberto and Pistilli, Francesca and Averta, Giuseppe},
  journal={arXiv preprint arXiv:2505.12911},
  year={2025}
}
```