# scNAME
### scNAME: neighborhood contrastive clustering with ancillary mask estimation for scRNA-seq data
Cell clustering is a vital procedure in scRNA-seq analysis, providing insight into complex biological phenomena. However, the noisy, high-dimensional
and large-scale nature of scRNA-seq data introduces challenges in clustering analysis. Up to now, many
deep learning-based methods have emerged to learn underlying feature representations while clustering.
However, these methods are inefficient when it comes to rare cell type identification and barely able to
fully utilize gene dependencies or cell similarity integrally. 
Here, we propose a novel scRNA-seq clustering algorithm called scNAME which incorporates a
mask estimation task for gene pertinence mining and a neighborhood contrastive learning framework for
cell intrinsic structure exploitation. The learned pattern through mask estimation helps reveal uncorrupted
data structure and denoise the original single-cell data. Additionally, the randomly created augmented
data introduced in contrastive learning not only helps improve robustness of clustering, but also increases
sample size in each cluster for better data capacity. Beyond this, we also introduce a neighborhood
contrastive paradigm with an offline memory bank, global in scope, which can inspire discriminative feature
representation and achieve intra-cluster compactness, yet inter-cluster separation.<br> 
<br>
![](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/38/6/10.1093_bioinformatics_btac011/2/btac011f1.jpeg?Expires=1689238224&Signature=JU~dy1dFNqhI1Ut9zaZzMzeSEGi9yAHPcl3Gtyqgu9IrrhDaP5J4tp51MsrKz0NopyLKmWePQmldUdde5ZKUcCU5X9of0Bz0mIHCwH2wamTrOGw66PZI60SV47Rn1uet2bTvSjb05xfAMfhsBsiQNO3IgtVetoZ-J3rEd-0bJL8EcT~~brPsmeBGDvD62ob8CmP93Vk1ywqUqzxb1l3JkmZ3f-1e7JWifNWwNj89FCz7P~r4CnxVRxLTfdFjNPUDBzn8veEXOEz7H-9VqCfsg2K40qAJBvBuFcogg~AQwkkEOr3x00UHt2ns4AcKKh~s5T0lzPk0u~StyggiUWr1~Q__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)
<br>
For more information, please refer to https://doi.org/10.1093/bioinformatics/btac011
## Requirement
python >= 3.8<br>
<br>
tensorflow-gpu >= 2.2.0<br>
<br>
Keras >= 2.3<br>
<br>
scanpy >= 1.9.1<br>
<br>
scikit-learn >= 0.22.2<br>
<br>
h5py >= 3.6.0<br>

## Reference
Wan, H., Chen, L., & Deng, M. (2022). scNAME: Neighborhood contrastive clustering with ancillary mask estimation for scRNA-seq data. Bioinformatics.<br>

If you have any questions, please contact: wanhui1997@pku.edu.cn


