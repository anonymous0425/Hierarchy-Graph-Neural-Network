
# Hierarchy graph neural network for jet classification in high-energy physics

This repository is the official implementation of [Hierarchy graph neural network for jet classificationin high-energy physics](https://arxiv.org/abs/2030.12345). 
![The architecture of HGN](https://github.com/anonymous0425/Hierarchy-Graph-Neural-Network/blob/master/overview.pdf)


## Requirements

The codebase is implemented in Python 3.7.4. package versions used for development are just below.
```
networkx          2.3
tqdm              4.36.1
numpy             1.17.2
pandas            0.25.1
scipy             1.3.1
torch             1.1.0
torch-scatter     1.3.1
torch-sparse      0.4.0
torch-cluster     1.4.4
torch-geometric   1.3.2
torchvision       0.3.0
scikit-learn      0.21.3
```

## Training

To train the model in the paper, run this command:

```train
python source/run.py --model 0 --processed_data paper --hid1 64 --hid2 128 --gpu 0 --batch_size 32 --exp_name train
```

## Evaluation

To evaluate my model, run:

```eval
python source/run.py --model 0 --processed_data paper --hid1 64 --hid2 128 --exp_name test --load saved_model/best_model.pkl --eval --gpu 0 --batch_size 64
```


## Pre-trained Models

The pretrained model can be found in the 'pretrained_model' folder.

## Results

Our model achieves the following performance on :

### [Jet Classification](https://www.biendata.com/competition/jet/data/)

| Model        | AUC  | 
| ------------------ |---------------- | 
|HGN  |    0.836         |    

