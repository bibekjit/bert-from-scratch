# bert-from-scratch

Implementation of bert from scratch

Pretrained on MLM task only

## The training is divided in 2 phases

1st phase is training short length sequences (maxlen = 60) and 2nd phase is for longer sequences (maxlen = 128)
phase 1 is for learning word embeddings and phase 2 for positional embeddings

this is done as the attention computation is quadratic, meaning it would cost more for (32, 128) compared to a (8, 512) input size
where 8 and 32 are batch size and 128 and 512 are maxlen. So, the stratergy is to train longer sequences for lets say 90k steps and remaining 10k for longer sequences
This way, it can capture contextual and sequential understanding at a lesser computational cost

In this model, 1st phase is trained for 230k+ steps and 2nd phase for 44k steps

## Tokenization

Unlike the original BERT paper, customized form of subword tokenization is used
that is, for "playing", instead of "play ##ing" it will be "play ## ing" 
this increases the sequence length but this will give consistency and also requires less vocab size 

