class Parameters :
    
    all_params = []
    
    def __init__(self , name , params) : 
        
        self.name = name
        self.params = params
        
        Parameters.all_params.append((self))
