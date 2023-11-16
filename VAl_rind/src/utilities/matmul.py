def matmul(f_row , f_col , s_row , s_col , funcs = None):
    
    if len(f_col) != len(s_row) : raise ValueError(f'Cannot multiply array with dimensions {len(f_row)}x{len(f_col)} , {len(s_row)}x{len(s_col)}')
    if funcs :
        
        if len(funcs) < len(f_col) : 
            
            warnings.warn(f'{len(funcs)} < {len(f_col)} : Defining the leftover to be Multiplications')
            
            funcs += [lambda x , y : x * y] * (len(s_col) - len(funcs))
        
        elif len(funcs) > len(f_col) : raise ValueError(f'{len(funcs)} > {len(f_col)} : Length of funcs should be <= Length of Matrix')
    else : funcs = [lambda x , y : x * y] * len(s_col)
            
    val = sum(list(map(
        lambda func , x , y : func(x , y) , funcs , f_col , s_row
    )))
    
    col = s_col * val / 2
    row = f_row * 2
    
    return row , col
