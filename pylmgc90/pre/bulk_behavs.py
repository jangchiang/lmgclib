from .shared import bulk_behav

class materials(dict):
    """ class materials:
        permet de stocker l'ensemble des materiaux
        
    """
    def addMaterial(self,*mat):
        """addMaterial(self,*mat)
        this function adds a material in the materials container

        parameters
         *mat the liste of materials
        """
        for m in mat:
            if not m.nom in list(self.keys()):
                assert isinstance(m, bulk_behav.material), "%r is not a material"%m
                self[m.nom] = m

    # surcharge de l'operateur +
    def __add__(self,ma):
       """__add__(self, ma):

       this function overrides the operator '+' to add a 
       material in the materials container

       parameters:

       - self: the materials container itself
       - ma: the material to add
       """
       self.addMaterial(ma)
       return self

                 
                
