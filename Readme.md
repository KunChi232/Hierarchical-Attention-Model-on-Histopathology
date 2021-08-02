## Hierarchical-Attention-Model


## Requirements
* Python==3.6.0
* torch==1.6.0
* torchvision==0.7.0
* sciki-learn
* numpy
* [smooth-topk](https://github.com/oval-group/smooth-topk)
* opencv-python
* tqdm

## Usage

Training & Validation:
``` python
python3 Train.py --level patient --hidden_dim 512 --encoder_layer 6 --k_sample 3 --tau 0.5 --save_path 'path/to/save/' --label 'path/to/label pickle file' --use_kather_data True --epoch 60 --lr 3e-4 --evaluate_mode kfold --kfold 5
```
TMA External Validation:
``` python
python3 TMA_Validation.py --level patient --hidden_dim 512 --encoder_layer 6 --k_sample 3 -- tau 0.5 --save_path 'path/to/saved/weights' --label 'path/to/label pickle file' --evaluate_mode kfold --kfold 5
```

## Whole slide images tiling
Please refer to this github [repo](https://github.com/mahmoodlab/CLAM), or you can download the preprocessed data provided by [Kather et al](https://www.nature.com/articles/s41591-019-0462-y).

## Data preparation

You can use any pre-trained CNN model to extract each patch's features. Organizing as following Python dictionary format and saving as pickle file.

Patch features pickle file will look like this:
``` python
{
  'patch_name' : array([latent feature]),
  'patch_name' : array([latent feature]),
  ...
}
```

Cluster pickle file:
``` python
{
  XXX_id: {
    'patch_name' : cluster label,
    'patch_name' : cluster label,
    ...
  },
  XXX_id: {
    'patch_name' : cluster label,
    'patch_name' : cluster label,
    ...
  },
}

```
Label pickle file:
``` python
{
  XXX_id: class,
  XXX_id: class,
  ...
}
```

Note: XXX_id can be patient's ID or slide's ID, which is depanding on your task. And please be sure that the patch_name in features pickle file and in cluster pickle file is the same.



