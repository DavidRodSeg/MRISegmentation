# MRISegmentation
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidRodSeg/MRISegmentation/blob/main/MRISegmentation_EnglishVersion.ipynb)
## 🎯 Objective

The aim of the project was to achieved binary segmentation of brain tumoral MRI (magnetic resonance imaging) images. For that purpose two models were used: pretrained DeepLabV3 and AGResUNet. Both use a _encoder-decoder_ arquitecture
and are fully convolutional networks. The quality of the models were measured using _dice_, _precision_, _recall_, _accuracy_ and _IoU_, typically use in this kind of problems.

## 📊 Data set analysis

The data set used consisted in a set of two-dimensional images of brain regions extracted through magnetic resonance imaging. We had several types of images or _sequences_ based on the parameters configured during the acquisition of these.

* **T1-weighted (T1)**: distinguishes between white and gray matter. Helps detect the presence of hemorrhages.
* **T2-weighted (T2)**: allows detection of regions with fluid, aiding in the detection of edemas and inflamed tissues.
* **T1-weighted with contrast (T1ce)**: highlights areas with a broken blood-brain barrier. Useful for identifying active tumor regions.
* **Fluid Attenuated Inversion Recovery (FLAIR)**: used to visualize areas with pathological changes such as edemas and specific areas of the tumor.
* **Binary segmentation**: labeling of pixels as tumor (1) or non-tumor (0) by experts.

![Alt Text](https://drive.google.com/uc?id=1yVNaf4qAEJFECK6N3N_YlseUHtJ22QDr)
<center>
<i>Figure 1: Example of the train set. (a) T1-weighted sequence, (b) T2-weighted sequence,
  (c) T1-weighted with contrast sequence, (d) FLAIR sequence, (e) Binary segmentation.</i>
</center>

The data set is divided in training and validation sets (994 images) y testing set (257 images). The resolution is 240 x 240 pixels..

## 🧪 Models
### DeepLabV3
The DeepLab models are a type of deep learning architecture dedicated to semantic image segmentation, that is, assigning a class to each pixel of an image. 
They are characterized by the use of dilated convolutions, which allow the model to increase its field of view without increasing the number of parameters by 
introducing gaps between the pixels of the convolution mask. Versions v2 and v3 of this model also include what is known as Atrous Spatial Pyramid Pooling (ASPP), 
which consists of applying different dilated convolutions with varying dilation rates, enabling the capture of information at different scales. It also includes what
is usually known as _encoder-decoder_ structure which consists in the union of two structures: an encoder, which encodes the image by extracting the most important features; and a decoder,
which allows obtaining the original image dimensions but from the features extracted by the encoder.

### AGResUNet
UNet models are a series of fully convolutional architectures with an encoder-decoder structure developed for biomedical image segmentation. 
They are characterized for using information from the encoder in the decoder through what are called _skip connections_. The encoder-decoder structure, 
along with these skip connections, gives the model its U-shape, from which it gets its name.

The AGResUNet model adds two additional elements to the structure:

* **Residual connections**: Typical of the ResNet model. The convolutional blocks of the UNet are replaced with residual blocks, which include information from the input image at the output of the block.
* **Attention gates**: Typical of the attention mechanisms of transformers. Attention gates are added during the UpSampling process of the decoder, allowing the network to focus on important regions of the MRI images.

![Alt Text](https://drive.google.com/uc?id=1oPHG1DFfsOFgbOAAYVM7E_h8NKoFYEw7)
<center>
<i>Figure 2: AGResUNet model. From: Attention Gate ResU-Net for Automatic
MRI Brain Tumor Segmentation. </i>
</center>

## 🛠️ How to use it
### 📦 Dependencies

The main libraries are:
* Pytorch = 2.5.1
* Numpy = 2.2.1

Additional dependencies are listed in the file  _requirement.txt_

### 🏋️ Training
Run _training.py_ to start training the models and visualizing the results.

### 📈 Results
For 20 epochs the metrics obtained for both models are the following:

| Models      | Accuracy | Precision | Recall | Dice  | IoU   |
|-------------|----------|-----------|--------|-------|-------|
| DeepLabV3   | 0.991    | 0.864     | 0.914  | 0.872 | 0.799 |
| AGResUNet   | 0.995    | 0.937     | 0.937  | 0.916 | 0.881 |

The results are specially satisfactory with AGResUNet. This is also shown in the following prediction:
![Alt Text](https://drive.google.com/uc?id=1ov7QdWR2qVUy_JWLgu5evEMnN2qrZvPL)
<center>
<i>Figure 3: AGResUNet model prediction. </i>
</center>


## 📚 References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-article3" class="csl-entry">

Oktay, Ozan, Jo Schlemper, Loic Folgoc, Matthew Lee, Mattias Heinrich,
Kazunari Misawa, Kensaku Mori, et al. 2018. “Attention u-Net: Learning
Where to Look for the Pancreas,” April.
<https://doi.org/10.48550/arXiv.1804.03999>.

</div>

<div id="ref-ronneberger2015unetconvolutionalnetworksbiomedical"
class="csl-entry">

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. 2015. “U-Net:
Convolutional Networks for Biomedical Image Segmentation.”
<https://arxiv.org/abs/1505.04597>.

</div>

<div id="ref-Siddique_2021" class="csl-entry">

Siddique, Nahian, Sidike Paheding, Colin P. Elkin, and Vijay
Devabhaktuni. 2021. “U-Net and Its Variants for Medical Image
Segmentation: A Review of Theory and Applications.” *IEEE Access* 9:
82031–57. <https://doi.org/10.1109/access.2021.3086020>.

</div>

<div id="ref-article" class="csl-entry">

Zhang, Jianxin, Zongkang Jiang, Jing Dong, Yaqing Hou, and Bin Liu.
2020. “Attention Gate ResU-Net for Automatic MRI Brain Tumor
Segmentation.” *IEEE Access* PP (March): 1–1.
<https://doi.org/10.1109/ACCESS.2020.2983075>.

</div>

<div id="ref-article4" class="csl-entry">

Zhang, Zhengxin, and Qingjie Liu. 2017. “Road Extraction by Deep
Residual u-Net.” *IEEE Geoscience and Remote Sensing Letters* PP
(November). <https://doi.org/10.1109/LGRS.2018.2802944>.

</div>

</div>


