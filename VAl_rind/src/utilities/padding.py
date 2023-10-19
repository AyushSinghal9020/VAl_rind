import numpy as np 

padding = lambda val , padding_length : np.concatenate([val , 
                                                        np.zeros(shape = padding_length - len(val) , 
                                                                 dtype = np.int8)])