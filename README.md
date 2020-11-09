# Category Prior Guided Mixup Data Argumentation for Charge Prediction

Source code of paper "Category Prior Guided Mixup Data Argumentation for Charge Prediction" submit to ACTA AUTOMATICA SINICA Journal.

## Dataset
We use dataset from paper "Few-Shot Charge Prediction with Discriminative Legal Attributes", https://github.com/thunlp/attribute_charge.

## Run
Run the following command to training model:

`
python train.py --tokenized --rnn_bidir --data_dir data/attribute_charge/data 
`

## Dependencies
```
torch~=1.5.1
numpy~=1.19.0
pytorch-ignite==0.4.2
tqdm==4.48.0
scikit-learn==0.23.1
```