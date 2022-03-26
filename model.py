import torch 
import torch.nn as nn 


class SegNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(SegNet, self).__init__()
        self.encoder = encoder 
        self.decoder = decoder 
        
    def forward(self, x):
        return x
    
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
    def forward(self, x):
        return x 
    
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
    def forward(self, x):
        return x 