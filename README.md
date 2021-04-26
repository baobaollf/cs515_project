# Spherical CNN using Wavelet Pooling in Classification


In this paper, we are going to apply a convolutional neural network with 3D spherical inputs and use Haar wavelet pooling as the pooling layer to show that we can achieve state-of-the-art accuracy, however, with training time increased significantly. 


## How to get started
- Download the dataset [[https://drive.google.com/file/d/1_yJCn0lWkb8gvaHxwTTkGlLVT8dKR0Hc/view?usp=sharing][here]] (7.0Gb).

- Save it to 
`drive/MyDrive/Colab\ Notebooks/m40_tf_csph_3d_48aug_64.tar.gz`

- Open file `code.ipydb` and follow the instructions. 



# References

Esteves, C., Allen-Blanchette, C., Makadia, A., & Daniilidis,
K. Learning SO(3) Equivariant Representations with Spherical
CNNs. European Conference on Computer Vision, ECCV 2018 (oral). http://arxiv.org/abs/1711.06721

``` bibtex
@article{esteves17_learn_so_equiv_repres_with_spher_cnns,
  author = {Esteves, Carlos and Allen-Blanchette, Christine and Makadia, Ameesh and Daniilidis, Kostas},
  title = {Learning SO(3) Equivariant Representations With Spherical Cnns},
  journal = {CoRR},
  year = {2017},
  url = {http://arxiv.org/abs/1711.06721},
  archivePrefix = {arXiv},
  eprint = {1711.06721},
  primaryClass = {cs.CV},
}
```

# Authors
Linfeng Li

[[http://machc.github.io][Carlos Esteves]] [1], [[http://www.seas.upenn.edu/~allec/][Christine Allen-Blanchette]] [1], [[http://www.ameeshmakadia.com][Ameesh Makadia]] [2], [[http://www.cis.upenn.edu/~kostas/][Kostas Daniilidis]] [1]

[1] [[http://grasp.upenn.edu][GRASP Laboratory]], [[http://www.upenn.edu][University of Pennsylvania]]

[2] [[http://research.google.com][Google]]
