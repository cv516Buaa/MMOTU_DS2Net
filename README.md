# MMOTU_DS<sup>2</sup>Net

This repo is the implementation of ["A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation"](https://arxiv.org/abs/2207.06799). we refer to  [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMGeneration](https://github.com/open-mmlab/mmgeneration) and mix them to implement unsupervised domain adaptation based segmentation (UDA SEG) task. Many thanks to SenseTime and their two excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig1.png" width = "100%" alt="MMOTU"/></td>
    <td><img src="PaperFigs\Fig4.png" width = "100%" alt="DS2Net"/></td>
    </tr>
</table>

## Dataset

**Multi-Modality Ovarian Tumor Ultrasound (MMOTU) image dataset** consists of two sub-sets with two modalities, which are **OTU\_2d** and **OTU\_CEUS** respectively including **1469 2d ultrasound images** and **170 CEUS images**. On both of these two sub-sets, we provide pixel-wise semantic annotations and global-wise category annotations.

**OTU\_2d** : [Data will soon be released after evaluated by Ethics Committee.]

**OTU\_CEUS**: [Data will soon be released after evaluated by Ethics Committee.]

## DS<sup>2</sup>Net

### Install

 	requirements:

​		python >= 3.7

​		pytorch >= 1.4

​		cuda >= 10.0

### Training

### Testing
