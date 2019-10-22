import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#_INF = float('inf')

class Attn(nn.Module):
    def __init__(self, method, dim):
        super(Attn, self).__init__()
        self.method = method
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim*2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, hidden, encoder_outputs):
        """
        hidden: 1 x batch_size x hidden_size (hidden_size = dim)
        Encoder_outputs: max_encoder_seq_len x batch_size x hidden_size
        Transfer to: 
        inputs: batch x dim
        context: batch x sourceL x dim
        """
        # Reshape data
        inputs = hidden.squeeze(0)
        context = encoder_outputs.transpose(0,1)

        targetT = self.linear_in(inputs).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        #if self.mask is not None:
        #    attn.data.masked_fill_(self.mask, -_INF)
        
        attn = F.softmax(attn, dim = 1) # batch x sourceL
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL
        #weightedContext = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        #contextCombined = torch.cat((weightedContext, inputs), 1) # batch_size  x 2*dim
        #contextOutput = self.tanh(self.linear_out(contextCombined))
        return attn3 #, contextOutput






        