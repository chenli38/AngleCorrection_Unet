# Illumination angle correction during image acquisition in light-sheet fluorescence microscopy using deep learning
A deep learning-based approach to estimate the angular error of the illumination beam relative to the detection focal plane. The illumination beam was then corrected using a pair of galvo scanners, and the correction significantly improved the image quality across the entire field-of-view. The angular estimation was based on calculating the defocus level on a pixel level within the image using two defocused images.  
Source code for paper (https://doi.org/10.1364/BOE.447392)  
A small test dataset is provided and free to download: https://doi.org/10.6084/m9.figshare.18515108  

## Required packages
pytorch, torchvision, matplotlib, numpy, random, skimage, scipy 

Most of the training scripts are inherited from https://github.com/chenli38/Defocus_classification.git  
## Angular and focus correction pipeline 
<img src="images/Figure1.png" width="458" height="712">  

## Function of scripts
**main.py**: 
 - loading dataset, network model and hyperparameters setting, training

**model.py**: 
 - define the Unet model structure

**utils.py**
 - generate the color bar (-36m to 36um for different color)

**generate_mask.py**
 - input: image directory
 - output: binary images based on the threshold.

**test_angle_file/.m**
 - predict the defocus distance of titled image dataset using Unet (testUnet.m), CNN (testPaper1.m), DCTS (testDCTS.m)
 - calculate the angle test_angle_Unet_yaw.m and test_angle_Unet_roll.m anc compared with ground truth. 

**label_angle_matlabfile**
 - a angle labelling tool developed by MATLAB appdesigner to help calcualte the roll and yaw angle of titled images

   
## Contact
cli38@ncsu.edu, ischen.li235@gmail.com

## References
- Li, Chen, et al. "Deep learning-based autofocus method enhances image quality in light-sheet fluorescence microscopy." Biomedical Optics Express 12.8 (2021): 5214-5226.
- Li, Chen, et al. "Illumination angle correction during image acquisition in light-sheet fluorescence microscopy using deep learning." Biomedical Optics Express 13.2 (2022): 888-901.
- Royer, Loïc A., et al. "Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms." Nature biotechnology 34.12 (2016): 1267-1278.
  
