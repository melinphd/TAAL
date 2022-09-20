# TAAL
Test-time Augmentation for Active Learning

A semi-supervised AL method identifying the most uncertain samples based on applied transformations.

The results were published in MICCAI_DALI 2022 workshop.


## Results
### Results
Model performance across different AL cycles (increasinig number of labeled samples, each identififed by given AL strategy).
![TAAL_results](https://user-images.githubusercontent.com/110574405/191175882-1d56181c-eca4-4eec-87f4-274f31696f7d.png)



## <ins> Data </ins> <br/>
Data folder has structure:
```
data
├── ACDC 
│ └── patient001 
│   └── Info.cfg 
│   └── patient001_4d.nii.gz 
│   └── patient001_frameXX.nii.gz 
│   └── patient001_frameXX_gt.nii.gz 
│   └── ...
│ └── ...
│ └── patient100 
│   └── Info.cfg 
│   └── patient001_4d.nii.gz 
│   └── patient001_frameXX.nii.gz 
│   └── patient001_frameXX_gt.nii.gz 
│   └── ..

```

Data taken from:  <br/>
- `ACDC Segmentation Challenge` (100 samples) (2D): https://acdc.creatis.insa-lyon.fr/ <br/>


### Data Preprocessing

ACDC dataset contains 3D MRI data from 100 patients. <br/>
The shapes of these volumes vary between [154, 428] in the x-direction, [154, 512] in the y-direction 
and [6, 18] in the z-direction.<br/>

We take the data for diastolic (ED) cardiac phase. <br/>

1) Select phase <br/>
We take the data for diastolic (ED) cardiac phase

2) Resample <br/>
We center the image and resample the spacing to 1.0 in the x- and y-direction. 
The new image width is 256 x 256. 
Spacing and length in the z-direction are kept the same.

3) Normalize <br/>
We apply 99th percentile normalization. 

4) Convert to 2D <br/>
We convert 3D volumes for each patient to 6-18 independent 2D slices. 
The slices are taken in the z-direction.


**Running Preprocessing (jupyter notebook):** `ACDC-preprocessing.ipynb`

Choose: 
- `num_test_patients = 20`
- `new_img_width = 256`
- `new_spacing = 1.`

This will randomly select 20 patients to be used for testing. Each slice is processed in order. <br/>

The preprocessed training data is saved in: `/data/preprocessed_ACDC_256_train_normalized`.
The preprocessed test data is saved in: `/data/preprocessed_ACDC_256_test_normalized`.

Here, patient ID list for testing is [81, 14, 3, 94, 35, 31, 28, 17, 13, 86, 69, 11, 75, 54, 4, 27, 29, 64, 77, 71]. <br/>
We create one file in each preprocessed data folder called `train_indices.txt` or `test_indices.txt`.
It contains 2 list with test patient ID's (`train indices` or `test indices`) and patient ID 
corresponding to each of the 191 test slices (`train_volume_list` or `test_volume_list`)
(used to compute the metric on the 3D volume).
Each folder also contains a file called `train_position_indices.txt` or `test_positoni_indices.txt` which 
contain a list of the relative position of each slice in its corresponding volume.

```
data
├── preprocessed_ACDC_256_train
│   └── data 
│   └── label 
|   └── train_indices.txt
|   └── train_position_indices.txt
├── preprocessed_ACDC_256_test
│   └── data 
│   └── label 
|   └── test_indices.txt
|   └── test_position_indices.txt

```
### Models
#### 1) Fully-supervised
- **solver** (fully supervised): 

#### 2) Semi-supervised

    
### Training the model
**Running Training:** `python main_train.py --device cuda:0 --config train_config`

`train_config.json` contains information on run name, training parameters and params on dataset. 

** Note that `Configs.configs' must be modified to add paths**

### Metrics
**Dice Score**: F1 score measuring the harmonic mean of precision and recall. 
It is a per-pixel detection score.