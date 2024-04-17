# Swin-U-NET for LocalBrainAge Prediction

This repository builds upon the Local UNet Brain Age Prediction by integrating Swin Transformer Blocks into both the Encoder and Decoder stages.

## Dependencies & Requirements

- Anaconda (Python 3.7)
- CUDA == 11.4
- GPU Memory >= 10 GB
- Memory >= 8 GB

```bash
conda create -n local-swin-unet python=3.7 -y
conda activate local-swin-unet
pip install -r requirements.txt
```

## Data Preparation

1. Download and install MATLAB and SPM12.
2. Run `bash spm12_preprocessing_pipeline.sh`.
   - Adjust the file path as per your dataset in line 20.
   - This command will call SPM12 to generate the gray matter and white matter in the root directory of the dataset.

## Training

Train a model by:

```bash
python3 full_training_script.py --num_encoding_layers=2 --num_filters=64 --num_subjects=2 --num_voxels_per_subject=2 --location_metadata=$path_metadata$ --dirpath_gm=$path_gm_data$ --dirpath_wm=$path_wm_data$ --dataset_name=$your_dataset_name$
```

- `--num_encoding_layers`: number of scales for UNET
- `--num_filters`: number of filters at each convolution operation
- `--location_metadata`: CSV file containing at least three columns: Subject, Age, and Gender
- `--dirpath_gm`: path to the processed gray matter directory
- `--dirpath_wm`: path to the processed white matter directory
- `--dataset_name`: name of your dataset

Parameters: More parameters can be found in the script.

Training weight: The training weight will be saved in the `./saved_model_3D_UNET_Dropout` directory.

Note that achieving the best possible performance in the training process can be quite time-consuming, typically taking around 2-3 weeks when utilizing a single GPU.

## Test

Test the model on your training dataset by:

```bash
python3 full_testing_script.py --filepath_csv=$path_test_metadata$ --dirpath_raw_data=$path_raw_T1_data$ --dataset_name=$your_dataset_name$ --size_batch_preprocessing=1
```

- `--filepath_csv`: CSV file for your test subjects
- `--dirpath_raw_data`: path to the directory containing the raw T1 nifti files.
- `--dataset_name`: name of your dataset
- `size_batch_preprocessing`:nifti files to process at the same time

## Pre-trained Model

The pre-trained model can be downloaded [Google Drive](https://drive.google.com/drive/folders/1yrHR_e86hJ9oAWBpaoZ5X14GKPfcWtwR?usp=sharing).

Please place the pre-trained model weights in the `saved_model_3D_UNET_Dropout/iteration_68000/` folder.

For data with custom preprocessing, we recommend training from scratch.

## Acknowledgment

This work is heavily reliant on the [U-NET-for-LocalBrainAge-prediction](https://github.com/SebastianPopescu/U-NET-for-LocalBrainAge-prediction).

## References

1. Popescu, Sebastian G., et al. "Local brain-age: a U-net model." Frontiers in Aging Neuroscience 13 (2021): 761954.
2. Liu, Ze, et al. "Video swin transformer." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
3. Cao, Hu, et al. "Swin-unet: Unet-like pure transformer for medical image segmentation." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.

