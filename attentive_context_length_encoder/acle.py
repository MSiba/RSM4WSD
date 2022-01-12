import torch
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct

embd_dim, num_heads, bsz = 10, 5, 64

in_proj_container = InProjContainer(torch.nn.Linear(embd_dim, embd_dim), torch.nn.Linear(embd_dim, embd_dim), torch.nn.Linear(embd_dim, embd_dim))

MHA = MultiheadAttentionContainer(nhead=num_heads,
                                  in_proj_container=in_proj_container,
                                  attention_layer=ScaledDotProduct(),
                                  out_proj=torch.nn.Linear(embd_dim, embd_dim),
                                  batch_first=False)


query = torch.rand((21, bsz, embd_dim))
key = value = torch.rand((16, bsz, embd_dim))
attn_output, attn_weights = MHA(query=query,
                                key=key,
                                value=value,
                                attn_mask=None,
                                bias_k=None,
                                bias_v=None)
print(attn_output.shape)
# torch.Size([21, 64, 10])

SDP = ScaledDotProduct(dropout=0.1,
                       batch_first=False)

#https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html


