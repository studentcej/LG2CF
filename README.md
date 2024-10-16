## Brief Introduction
 This paper has investigated the potential of exploring and exploiting latent group information to improve recommendation performance for implicit collaborative filtering.
## Prerequisites
- Python 3.8 
- PyTorch 1.11.0

## Some Tips
Flags in `parse.py`:

Model training related settings:

- `--train_mode` Choosing to either start a new training session, or continue training with the model saved from your previous session.
- `--epochs` Number of sweeps over the dataset to train.
- `--dataset` Choosing 100k, 1M or your dataset.

You can set the relevant parameters for model training,

- `--batch_size` size of each batch
- `--l2` l2 regulation constant.
- `--lr` learning rate.
- `--lr_dc` learning rate decay rate.
- `--lr_dc_epoch` training epoch at which the learning rate starts to decay.
- `--N` number of samples for negative sampling

#### Suggested Model Training Parameters
|                    | batch_size |   l2    |  lr  | lr_dc | lr_dc_epoch | dim  | 
|--------------------|:----------:|:-------:|:----:|:-----:|:-----------:|------|
| 100k-MF            |    1024    |  1e-5   | 5e-4 |   1   |     []      | 32   |
| 1M-MF              |    1024    |  1e-5   | 5e-4 |   1   |     []      | 128  |
| Gowalla-MF         |    1024    |    0    | 5e-4 |   1   |     []      | 1024 |
| Yelp2018-MF        |    1024    |  1e-6   | 5e-4 |   1   |     []      | 2048 |


LG2CF related parameters:
- `--alpha` is used to control the proportion of integrating latent group information to update the original user/item embedding
- `--N_p` to the user(item) embeddings in a two-tower model, and the 𝑁𝑝 is used to control when to activate the exploring and exploiting process in the pre-training phase. We fixed 𝑁𝑝=50 activate the LG2CF module in our comparison studies.


#### Suggested α Parameters
|                   |    α    |
|-------------------|:-------:|
| 100k-MF           |  0.987  |
| 1M-MF             |  0.972  |
| Gowalla-MF        |  0.966  |
| Yelp2018-MF       | 0.99999 |


For instance, execute the following command to train ICF model using LG2CF module.
```
python main.py  --dataset 100k --l2 1e-5 --lr 5e-4 --dim 32  --batch_size 1024 --alpha 0.987 --N_p 50 --epochs 1000
python main.py  --dataset 1M   --l2 1e-5 --lr 5e-4 --dim 128 --batch_size 1024 --alpha 0.972 --N_p 50 --epochs 1000
```
