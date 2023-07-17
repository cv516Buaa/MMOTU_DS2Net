# MMOTU_DS<sup>2</sup>Net

This repo is the implementation of ["A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation"](https://arxiv.org/abs/2207.06799). we refer to  [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMGeneration](https://github.com/open-mmlab/mmgeneration) and mix them to implement unsupervised domain adaptation based segmentation (UDA SEG) task. Many thanks to SenseTime and their two excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig1.png" width = "100%" alt="MMOTU"/></td>
    <td><img src="PaperFigs\Fig4.png" width = "100%" alt="DS2Net"/></td>
    </tr>
</table>

## Dataset

**Multi-Modality Ovarian Tumor Ultrasound (MMOTU) image dataset** consists of two sub-sets with two modalities, which are **OTU\_2d** and **OTU\_CEUS** respectively including **1469 2d ultrasound images** and **170 CEUS images**. On both of these two sub-sets, we provide pixel-wise semantic annotations and global-wise category annotations. **Many thanks to Department of Gynecology and Obstetrics, Beijing Shijitan Hospital, Capital Medical University and their excellent works on collecting and annotating the data.**

**MMOTU** : [google drive](https://drive.google.com/drive/folders/1c5n0fVKrM9-SZE1kacTXPt1pt844iAs1?usp=sharing) (move OTU_2d and OTU_3d to data folder. Here, OTU_3d folder indicates OTU_CEUS in paper.)

## DS<sup>2</sup>Net

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.4
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html); Please don't forget to install mmsegmentation with

     ```
     cd MMOTU_DS2Net
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training

**mit_b5.pth** : [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing) (Before training Segformer or DS<sup>2</sup>Net_T, loading ImageNet-pretrained mit_b5.pth is very useful. We provide this pretrained backbone here. The pretrained backbone has already been transformed to fit for our repo.)

#### Task1: Single-modality semantic segmentation

<table>
    <tr>
    <td><img src="PaperFigs\SSeg.jpg" width = "100%" alt="Single-Modality semantic segmentation"/></td>
    </tr>
</table>
  
     cd MMOTU_DS2Net
     
     ./tools/dist_train.sh ./experiments/pspnet_r50-d8_769x769_20k_MMOTU/config/pspnet_r50-d8_769x769_20k_MMOTU.py 2

#### Task2: UDA semantic segmentation

<table>
    <tr>
    <td><img src="PaperFigs\UDASeg.jpg" width = "100%" alt="UDA Multi-Modality semantic segmentation"/></td>
    </tr>
</table>

     cd MMOTU_DS2Net
     
     ./tools/dist_train.sh ./experiments/DS2Net_segformerb5_769x769_40k_MMOTU/config/DS2Net_segformerb5_769x769_40k_MMOTU.py 2

#### Task3: Single-modality recognition: 

<table>
    <tr>
    <td><img src="PaperFigs\SCls.jpg" width = "100%" alt="Single-Modality recognition"/></td>
    </tr>
</table>

### Testing

#### Task1: Single-modality semantic segmentation
  
     cd MMOTU_DS2Net
     
     ./tools/dist_test.sh ./experiments/pspnet_r50-d8_769x769_20k_MMOTU/config/pspnet_r50-d8_769x769_20k_MMOTU.py ./experiments/pspnet_r50-d8_769x769_20k_MMOTU/results/iter_80000.pth --eval mIoU

#### Task2: UDA semantic segmentation

     cd MMOTU_DS2Net
     
     ./tools/dist_test.sh ./experiments/DS2Net_segformerb5_769x769_40k_MMOTU/config/DS2Net_segformerb5_769x769_40k_MMOTU.py ./experiments/DS2Net_segformerb5_769x769_40k_MMOTU/results/iter_40000.pth --eval mIoU

### Generlization Experiments on [WHS-MR_CT](https://github.com/FupingWu90/CT_MR_2D_Dataset_DA): UDA semantic segmentation

     #### use ./tools/convert_datasets/WHS2d.sh to convert dataFolder for our repo!
     #### copy dataset to ./data
     cd MMOTU_DS2Net

     #### Training
     ./tools/dist_train.sh ./experiments/DS2Net_segformerb5_40k_WHS/config/DS2Net_segformerb5_40k_WHS_MR2CT.py 2
     #### Testing
     ./tools/dist_test.sh ./experiments/DS2Net_segformerb5_40k_WHS/config/DS2Net_segformerb5_40k_WHS_MR2CT.py ./experiments/DS2Net_segformerb5_40k_WHS/results/MR2CT_iter_3200_81.11.pth 2 --eval mDice

## Description of MMOTU/DS<sup>2</sup>Net
- https://arxiv.org/abs/2207.06799 

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

If you find this code useful please cite:
```
@article{DBLP:journals/corr/abs-2207-06799,
  author    = {Qi Zhao and
               Shuchang Lyu and
               Wenpei Bai and
               Linghan Cai and
               Binghao Liu and
               Meijing Wu and
               Xiubo Sang and
               Min Yang and
               Lijiang Chen},
  title     = {A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised
               Cross-Domain Semantic Segmentation},
  journal   = {CoRR},
  volume    = {abs/2207.06799},
  year      = {2022},
}
```

# References
Many thanks to their excellent works
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [MMGeneration](https://github.com/open-mmlab/mmgeneration)
