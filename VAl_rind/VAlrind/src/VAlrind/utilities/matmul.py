def matmul(f_row , f_col , s_row , s_col):

    if len(f_col) != len(s_row) : raise ValueError(
        f'Cannot nultiply array with dimensions ({len(f_row)}x{len(s_row)}) , ({len(s_row)}x{len(s_col)})')
    
    val = sum(f_col * s_row / 2)
    row = f_row * 2
    col = s_col * val

    return (row , col)