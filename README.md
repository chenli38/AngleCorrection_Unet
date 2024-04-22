# Illumination angle correction during image acquisition in light-sheet fluorescence microscopy using deep learning
A deep learning-based approach to estimate the angular error of the illumination beam relative to the detection focal plane. The illumination beam was then corrected using a pair of galvo scanners, and the correction significantly improved the image quality across the entire field-of-view. The angular estimation was based on calculating the defocus level on a pixel level within the image using two defocused images.  
Source code for paper (https://doi.org/10.1364/BOE.447392)  
A small test dataset is provided and free to download: https://doi.org/10.6084/m9.figshare.18515108  

## Required packages
pytorch, torchvision, matplotlib, numpy, random, skimage, scipy 

Most of the training scripts are inherited from https://github.com/chenli38/Defocus_classification.git  

## Function of scipts
**main.py**: 
 - loading dataset, network model and hyperparameters setting, training

**model.py**: 
 - define the Unet model structure

**utils.py**
 - generate the color bar (-36m to 36um for different color)

**generate_mask.py**
 - input: image directory
 - output: binary images depends on the threshold.
   

## Contact
cli38@ncsu.edu

## References
- Li, Chen, et al. "Deep learning-based autofocus method enhances image quality in light-sheet fluorescence microscopy." Biomedical Optics Express 12.8 (2021): 5214-5226.
- Li, Chen, et al. "Illumination angle correction during image acquisition in light-sheet fluorescence microscopy using deep learning." Biomedical Optics Express 13.2 (2022): 888-901.
- Royer, Loïc A., et al. "Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms." Nature biotechnology 34.12 (2016): 1267-1278.
  
