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
![](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/38/6/10.1093_bioinformatics_btac011/1/btac011f1.jpeg?Expires=1657506771&Signature=W34JD-5LnUyLl6rVyIBrKlX95WGTvYywCz~0G243VnUuPekZZuYuYrg8xmRVusmj6vWpZStp8eEWpJXI3KBGM8GFVuyJR-6Fek3Px6dsSIWHt69t~5Zrp27czV7h0BAQz4tkVVAQPyHqdF4IJCIl7yNduNoNHvm0M~m2jw1N0lLHhC80ueYCq5M5peQfyDuwdfDdFX5heWF4hnBRdZ3PTOLpNTu~A8pTaOfnieVrerGqp0eQgVq-yZM5dhhdz4oBb6swNyni7bNVApuM1IpdMnqcF~PT0-RrGOHwijugHkvs0NqA4xRBdxJOjSa6NpWHPG9MtGOiPbDdx-S6GbzdSg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)
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


