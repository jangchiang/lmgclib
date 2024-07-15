# module fournissant des macros de depot dans des conteneurs predifnis

# import du module permettant de savoir si on pourra importer les pre_tools
from ...utilities.check_compiled_modules import *

# import des des wrappers des anciens pre-processeurs pour les milieux granualaires

# si on peut essayer d'importer le module pre_tools sans tout faire planter
if import_lmgc90():
   # on essaye
   try:
      from ....chipy import lmgc90
   except:
      raise

# fonction qui depose les particules dans une boite paralepipedique
def depositInBox3D(radii, lx, ly, lz, deposited_radii=None, deposited_coor=None, seed=None):
   '''[nb_remaining_particles, coor]=depositInBox3D(radii, lx, ly, lz, deposited_radii=None, deposited_coor=None):

   this function deposits spherical particles in a box

   parameters:

   - radii: radii of the particles
   - lx: width of the box, following Ox axis
   - ly: width of the box, following Oy axis
   - lz: heigth of the box

   N.B. a point (x, y, z) is in the box iff x is in [-lx/2, lx/2], y is in [-ly/2, ly/2] and z is in [0, lz]

   optional parameters:

   - deposited_radii=None: radii of particles supposed to be already deposited
   - deposited_coor=None: coordinates of these deposited particles  
   - seed=None: an input seed to control randomness

   returned values:

   - nb_remaining_particles: number of deposited particles
   - coor: coordinates of the deposited particles [x1, y1, z1, x2, y2, z2, ...]

   WARNING: this function changes the radii list since it nullifies radii of the 
   particles out of the box. So, after calling this function, only the 
   first nb_remaining_particles particles are of interest''' 
  
   # on recupere le nombre de grains
   nb_particles=len(radii)

   # on depose les grains sous gravite dans la boite :
   # l'appel varie en fonction de la presence ou non de particules deja deposees
   if deposited_radii is None and deposited_coor is None: # cas sans particules
      if seed is not None:
        nb_comp_particles, coor=lmgc90.deposit3D_Box(radii, lx, ly, lz, 3*nb_particles, seed)
      else:
        nb_comp_particles, coor=lmgc90.deposit3D_Box(radii, lx, ly, lz, 3*nb_particles)

   elif deposited_radii != None and deposited_coor != None: # cas avec particules
      # deja deposees   
      if seed is not None:
        nb_comp_particles, coor=lmgc90.deposit3D_HeterogeneousBox(radii, lx, ly, lz, 
                                                                  deposited_radii,
                                                                  deposited_coor,
                                                                  3*nb_particles, seed)
      else:
        nb_comp_particles, coor=lmgc90.deposit3D_HeterogeneousBox(radii, lx, ly, lz, 
                                                                  deposited_radii,
                                                                  deposited_coor,
                                                                  3*nb_particles)

   else: # cas mal defini
      showError('to compute a deposit involving already deposited particles, radii AND coordinates of the deposited particles!')
     
   # on renvoie le nombre de particules deposees par l'algorithme et leur
   # coordonnees
   return [nb_comp_particles, coor]

# fonction qui depose les particules dans une boite cylindrique
def depositInCylinder3D(radii, R, lz, deposited_radii=None, deposited_coor=None, seed=None):
   '''[nb_remaining_particles, coor]=depositInCylinder3D(radii, R, lz, deposited_radii=None, deposited_coor=None, seed=None):

   this function deposits spherical particles in a cylinder

   parameters:

   - radii: radii of the particles
   - R: radius of the cylinder
   - lz: heigth of the cylinder

   N.B. a point (x, y, z) is in the cylinder iff x^2 + y^2 is in [0, R^2] and z is in [0, lz]

   optional parameters:

   - deposited_radii=None: radii of particles supposed to be already deposited
   - deposited_coor=None: coordinates of these deposited particles  
   - seed=None: an input seed to control randomness

   returned values:

   - nb_remaining_particles: number of deposited particles
   - coor: coordinates of the deposited particles [x1, y1, z1, x2, y2, z2, ...]

   WARNING: this function changes the radii list since it nullifies radii of the 
   particles out of the box. So, after calling this function, only the 
   first nb_remaining_particles particles are of interest''' 
  
   # on recupere le nombre de grains
   nb_particles=len(radii)
   # on depose les grains sous gravite dans le cylindre :
   # l'appel varie en fonction de la presence ou non de particules deja deposees
   if deposited_radii is None and deposited_coor is None: # cas sans particules
      if seed is not None:
        nb_comp_particles, coor=lmgc90.deposit3D_Cylinder(radii, R, lz, 3*nb_particles, seed)
      else:
        nb_comp_particles, coor=lmgc90.deposit3D_Cylinder(radii, R, lz, 3*nb_particles)

   elif deposited_radii != None and deposited_coor != None: # cas avec particules
      # deja deposees   
      if seed is not None:
        nb_comp_particles, coor=lmgc90.deposit3D_HeterogeneousCylinder(radii, R, lz, 
                                                                       deposited_radii,
                                                                       deposited_coor,
                                                                       3*nb_particles, seed)
      else:
        nb_comp_particles, coor=lmgc90.deposit3D_HeterogeneousCylinder(radii, R, lz, 
                                                                       deposited_radii,
                                                                       deposited_coor,
                                                                       3*nb_particles)
   else: # cas mal defini
      showError('to compute a deposit involving already deposited particles, radii AND coordinates of the deposited particles!')
     
   # on renvoie le nombre de particules deposees par l'algorithme et leur
   # coordonnees
   return [nb_comp_particles, coor]

# fonction qui depose les particules dans une boite spherique
def depositInSphere3D(radii, R, center, deposited_radii=None, deposited_coor=None, seed=None):
   '''[nb_remaining_particles, coor]=depositInSphere3D(radii, R, center, deposited_radii=None, deposited_coor=None, seed=None):

   this function deposits spherical particles in a cylinder

   parameters:

   - radii: radii of the particles
   - R: radius of the sphere
   - center: center of the sphere

   N.B. a point (x, y, z) is in the sphere iff (x - x_C)^2 + (y - y_C)^2 + (z - z_C)^2 is in [0, R^2]

   optional parameters:

   - deposited_radii=None: radii of particles supposed to be already deposited
   - deposited_coor=None: coordinates of these deposited particles
   - seed=None: an input seed to control randomness

   returned values:

   - nb_remaining_particles: number of deposited particles
   - coor: coordinates of the deposited particles [x1, y1, z1, x2, y2, z2, ...]

   WARNING: this function changes the radii list since it nullifies radii of the 
   particles out of the box. So, after calling this function, only the 
   first nb_remaining_particles particles are of interest''' 
  
   # on recupere le nombre de grains
   nb_particles=len(radii)

   # on depose les grains sous gravite dans la sphere :
   # l'appel varie en fonction de la presence ou non de particules deja deposees
   if deposited_radii is None and deposited_coor is None: # cas sans particules
      if seed is not None:
        nb_comp_particles, coor=lmgc90.deposit3D_Sphere(radii, R, center, 3*nb_particles, seed)
      else:
        nb_comp_particles, coor=lmgc90.deposit3D_Sphere(radii, R, center, 3*nb_particles)

   elif deposited_radii != None and deposited_coor != None: # cas avec particules
      # deja deposees   
      if seed is not None:
        nb_comp_particles, coor=lmgc90.deposit3D_HeterogeneousSphere(radii, R, center, 
                                                                     deposited_radii,
                                                                     deposited_coor,
                                                                     3*nb_particles, seed)
      else:
        nb_comp_particles, coor=lmgc90.deposit3D_HeterogeneousSphere(radii, R, center, 
                                                                     deposited_radii,
                                                                     deposited_coor,
                                                                     3*nb_particles)

   else: # cas mal defini
      showError('to compute a deposit involving already deposited particles, radii AND coordinates of the deposited particles!')
     
   # on renvoie le nombre de particules deposees par l'algorithme et leur
   # coordonnees
   return [nb_comp_particles, coor]
 
