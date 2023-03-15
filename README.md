## DeepPCC: Learned Lossy Point Cloud Compression
### Code will be available soon

### Abstract
We propose the DeepPCC, an end-to-end learning-based approach for the lossy compression of large-scale object point clouds.  For both geometry and attribute components, we introduce the Multiscale Neighborhood Information Aggregation (NIA) mechanism, which applies resolution downscaling progressively (i.e., dyadic downsampling of geometry and  average pooling of attribute) and combines sparse convolution and local self-attention at each resolution scale for effective feature representation. Under a simple autoencoder structure,  scale-wise NIA blocks are stacked as the analysis and synthesis transform in the encoder-decoder pair to best characterize spatial neighbors for accurate probability approximation of geometry occupancy and attribute intensity for point cloud compression (PCC).  Experiments demonstrate that the DeepPCC remarkably outperforms state-of-the-art rules-based MPEG G-PCC and learning-based JPEG VM both quantitatively and qualitatively, evidencing that DeepPCC is a promising solution for emerging AI-based PCC. The source code will be made publicly accessible soon.

### Examples

![Examples](images/visual_front.png)

### Method

![Diagram](images/overall_framework.png)


#### Dependencies
*Coming soon*

#### Training

*Coming Soon*

#### Results

*Coming Soon*

#### Citation

*Coming Soon*
