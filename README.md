# Python and Bash code used for Analaysis


Summary: Raw code for banded ridge regression analysis of Partly Cloudy fMRI data

Hi! This is the code that was used in our paper, "Early neural development of social perception: evidence from voxel-wise encoding in young children and adults" (Im, Shirahatti, Isik. preprint)

To run this code, you will need to get the video stimulus from Richardson et al (2018) and fMRI data from Hillary Richardson's open fMRI page, download tikreg and PyMoten packages from Gallant Lab Github (https://github.com/gallantlab).
To run ROI analyses, you would also need to download the ROI masks we used: We used STS mask from Ben Deen's group (https://bendeen.com/data/) and MT from Sabine Kastner's group (https://napl.scholar.princeton.edu/resources)

Note on movie and fMRI segmentation:
The movie was segmented into 2s segments (corresponding to TRs) and trimmed to remove the opening (segments 0-4) and closing credits (segments 157-174).
The fMRI data was also trimmed to remove the opening (TRs 0–10) and closing (TRs 162–168), as well as periods of rest before and after the film. 
For more information on how we trim the movie data please look at “trim_blank_tr” function in the "data_prep_helpers.py" file. 

# How to use the code
Broadly, there's three types of files: 
1) Folder 'data_prep' gives you files needed for data prepping like, isc masking or feature labeling csv for correlation and PCA.
2) Folder 'rockfish_scripts' has the codes for bash and python scripts used to run the encoding model, difference analysis between age groups, and thrsholding
3) Folder 'jupyter_notebooks' has the notebooks we used to visualize the data we analyzed using rockfish_scripts
4) Folder 'demo_sample' gives you example of how to use the notebooks
5) We also have helper files (ends with .py) where you can find the functions we used to run the codes


More specifically, to run the code, you would first want to run the dat_prep folder. Then run the rockfish_scripts folder. The workflow order should be Encoding Analysis & Differences -> Generate Null -> Threshold. After running the encoding analysis and thresholding data, you can run jupyter_notebooks folder. We used figures.ipynb for whole brain analysis and ROI_indv_social_features (individual social feature model prediction), ROI_moten_all_ages (motion energy prediction), and ROI_social_all_ages (social feature combined model prediction) for ROI analyses.

