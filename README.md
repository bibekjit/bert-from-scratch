# bert-from-scratch

Implementation of bert from scratch

Pretrained on MLM task only

## The training is divided in 2 phases

1st phase is training short length sequences (maxlen = 60) and 2nd phase is for longer sequences (maxlen = 128)
phase 1 is for learning word embeddings and phase 2 for positional embeddings

this is done as the attention computation is quadratic, meaning it would cost more for (32, 128) compared to a (8, 512) input size
where 8 and 32 are batch size and 128 and 512 are maxlen. 

In this model, 
