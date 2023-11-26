class Parameter :
    
    all_params = []
    
    def __init__(self , name , params) : 
        
        self.name = name
        self.params = params
        
        Parameter.all_params.append((self))
