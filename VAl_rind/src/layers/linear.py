from utilities import matmul , Parameters
import numpy as np

class linear: 

    def __init__(self , in_feats , out_feats):

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.in_col = np.random.uniform(0.01 , 0.001 , self.in_feats)
        self.out_col = np.random.uniform(0.01 , 0.001 , self.out_feats)


        self.params = [self.in_col , self.out_col] 
        
        self.params = [Parameters('Linear' , val) for val in self.params]
        
    def forward(self , inps):

        if len(inps.shape) == 1 : # 1D
            return_val = matmul(self.in_col , inps , 
                                self.in_col , self.out_col)
        
        elif len(inps.shape) == 2 : # 2D 
            return_val = np.vstack([matmul(self.in_col , val , 
                                           self.in_col , self.out_col)
                                    for val in inps])

        elif len(inps.shape) == 3 : # Batch
            return_val = []

            for batch in inps:

                batched = np.vstack([matmul(self.in_col , val , 
                                            self.in_col , self.out_col)
                                     for val in batch])
            
                return_val.append(batched)

            return_val = np.stack(return_val)

        else : raise ValueError(f'Inputs of shape {inps.shape} cannot be sent to processed')

        return return_val
