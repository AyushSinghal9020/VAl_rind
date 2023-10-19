from layers import mhsa , linear
from utilities import layer_norm

class Encoder_Block:

    def __init__(self , model_dim):

        self.model_dim = model_dim
        
        self.mhsa = mhsa(self.model_dim)
        self.linear_ = linear(self.model_dim , self.model_dim)

        self.parameters = [self.mhsa.parameters]

    def forward(self , inps):

        attention , weights = self.mhsa.forward(inps , inps, inps)

        attention = layer_norm(attention)

        linear_attention = self.linear_.forward(attention)
        attention = layer_norm(linear_attention + attention)

        return attention , weights