# module dedie au depot de particules sur un reseau

import numpy

# fonction qui genere une liste de coordonnees sur un resau cubique
# parametres :
#   - nb_ele_x : nombre de particules sur la premiere couche, dans la direction
#         x
#   - nb_ele_y : nombre de particules sur la premiere couche, dans la direction
#         y
#   - nb_layer : nombre de couches
#   - l : longueur d'un element du reseau, i.e. distance entre deux centres de
#         particules
# N.B. : le nombre total de positions generees est : 
#           nb_ele_x * nb_ele_y * nb_layer
# parametres optionnels :
#   - (x0, y0, z0) : coordonnees du coin inferieur gauche de la boite a remplir
#                i.e. la premiere position genere est 
#                (x0 + l/2, y0 + l/2, z0 + l/2)
# valeur de retour :
#   - vecteur des coordonnees [x1, y1, z1, x2, y2, z2, ...]
# ATTENTION : 
#    1. les particules a deposer sur le reseau doivent verifier : 
#          max(rayons) <= l/2
#    2. l'ensemble particules deposees sur ce resau est inclus dans une boite 
#       rectangulaire de dimensions : nb_ele_x*l x nb_ele_y*l x nb_layer*l
def cubicLattice3D(nb_ele_x, nb_ele_y, nb_layer, l, x0=0., y0=0., z0=0.):
   '''coor=cubicLattice3D(nb_ele_x, nb_ele_y, nb_layer, l, x0=0., y0=0., z0=0.):

  this function compute a list of positions on a cubic lattice

  parameters:

  - nb_ele_x: number of particles on the first layer, following axis (Ox) (the lowest)
  - nb_ele_z: number of particles on the first layer, following axis (Oz) (the lowest)
  - nb_layer: number of layers
  - l: length of a lattice element, i.e. distance between two 
    consecutive positions on the same layer, or the same column

  optional parameters:

  - (x0, y0, z0): position of the lower left corner of the bounding box 
    of the lattice, i.e. the first position is (x0 + l/2, y0 + l/2, z0 + l/2)

  return value:

  - coordinates of the positions [x1, y1, z1, x2, y2, z2, ...]

  N.B.: the total number of positions is nb_ele_x*nb_ele_y*nb_layer

  WARNING:

  1. the maximal radius of the particles to be deposited max_radius must 
     verify: max_radius <= l/2
  2. the dimensions of the bounding box of the lattice are :
     nb_ele_x*l x nb_ele_y x nb_layer*l'''

   # on initialise le vecteur qui va recevoir les coordonnees
   coor = numpy.zeros(3*nb_ele_x*nb_ele_y*nb_layer, 'd')
   # on construit la liste de positions
   for k in range(0, nb_layer, 1):
      for j in range(0, nb_ele_y, 1):
         for i in range(0, nb_ele_x, 1):
            # abscisse du point courant
            coor[3*(k*nb_ele_y*nb_ele_x + j*nb_ele_x + i)] = x0 + (i + 0.5)*l
            # ordonnee du point courant
            coor[3*(k*nb_ele_y*nb_ele_x + j*nb_ele_x + i) + 1] = y0 + (j + 0.5)*l
            # cote du point courant
            coor[3*(k*nb_ele_y*nb_ele_x + j*nb_ele_x + i) + 2] = z0 + (k + 0.5)*l

   # on renvoie la liste de coordonnees generee
   return coor

