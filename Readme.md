## Requirements
* Pyhon==3.6.0
* torch==1.6.0
* torchvision==0.7.0
* sciki-learn
* numpy
* smooth-topk
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


