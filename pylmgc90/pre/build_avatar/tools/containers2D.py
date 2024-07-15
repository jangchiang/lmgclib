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

import numpy
import math

# ATTENTION : radii peut etret modifie suite a l'appel a ces fonctions!

# fonction qui depose les particules dans une boite rectangulaire
def depositInBox2D(radii, lx, ly, deposited_radii=None, deposited_coor=None):
   '''[nb_remaining_particles, coor]=depositInBox2D(radii, lx, ly, deposited_radii=None, deposited_coor=None):

   this function deposits circular particles in a box

   parameters:

   - radii: radii of the particles
   - lx: length of the box
   - ly: heigth of the box

   optional parameters:

   - deposited_radii=None: radii of particles supposed to be in the box before the deposit
   - deposited_coor=None: radii of these deposited particles  

   returned values:

   - nb_remaining_particles: number of deposited particles
   - coor: coordinates of the deposited particles [x1, y1, x2, y2, ...]

   WARNING: 

   1. this function changes the radii list since it nullifies radii of the 
      particles out of the box. So, after calling this function, only the 
      first nb_remaining_particles particles are of interest 
   2. to avoid interpenetrations between particles and walls, this function
      uses a shrink based on the size of the particles'''
  
   # on recupere le nombre de grains
   nb_particles=len(radii)
   # on recupere le rayon du plus gros grain
   radius_max=max(radii)

   # on depose les grains sous gravite

   # si on a donne une liste de particules deja presentes dans la boite
   if (deposited_radii != None and deposited_coor != None):
      # on utilise la methode qui en tient compte
      coor=lmgc90.deposit2D_GravityAndBigParticles(radii, lx, 
         deposited_radii, deposited_coor, 2*nb_particles)
   # sinon,
   else:
      # on utilise la methode de base 
      coor=lmgc90.deposit2D_Gravity(radii, lx, 2*nb_particles)

   # on definit la polyligne adaptee pour une boite : une ligne horizontale
   # en y=ly - radius_max 
   slope_coor=[-0.5*lx, ly - radius_max, 1.5*lx, ly - radius_max]
   # N.B.: on definit dans une ligne legerement plus basse, pour eliminer les
   #       les interpentrations avec le bord superireur de la boite
   # on enleve les grains au-dessus de la ligne
   nb_remaining_particles=lmgc90.cut2D_Cut(radii, coor, slope_coor)
   
   # on renvoie le nombre de prticules restantes et les coordonnees des 
   # particules
   return [nb_remaining_particles, coor]

# fonction qui depose les particules dans un dique
def depositInDisk2D(radii, r, deposited_radii=None, deposited_coor=None):
   '''[nb_remaining_particles, coor]=depositInDisk2D(radii, r, deposited_radii=None, deposited_coor=None):

   this function deposits circular particles in a circular container

   parameters:

   - radii: radii of the particles
   - r: radius of the container 

   optional parameters:

   - deposited_radii=None: radii of particles supposed to be in the box before the deposit
   - deposited_coor=None: radii of these deposited particles  

   returned values:

   - nb_remaining_particles: number of deposited particles
   - coor: coordinates of the deposited particles [x1, y1, x2, y2, ...]

   WARNING: this function changes the radii list since it nullifies radii of 
   the particles out of the box. So, after calling this function, only
   the first nb_remaining_particles particles are of interest'''

   # on recupere le nombre de grains
   nb_particles=len(radii)

   # on realise un depot sous gravite, dans une boite de largueur 2.*r

   # si on a donne une liste de particules deja presentes dans la boite
   if (deposited_radii != None and deposited_coor != None):
      # on utilise la methode qui en tient compte
      coor=lmgc90.deposit2D_GravityAndBigParticles(radii, 2.*r,
         deposited_radii, deposited_coor, 2*nb_particles)
   # sinon,
   else:
      # on utilise la methode de base 
      coor=lmgc90.deposit2D_Gravity(radii, 2.*r, 2*nb_particles)

   # on definit un contour circulaire, de rayon r et centre en [r, r]
   slope_coor=numpy.zeros(160, 'd')
   for i in range(0, 80, 1):
      slope_coor[2*i]=r + r*math.cos(math.pi*(1. - i*0.025))
      slope_coor[2*i + 1]=r + r*math.sin(math.pi*(1. - i*0.025))
   # on enleve les grains hors du contour
   nb_remaining_particles=lmgc90.cut2D_Cut(radii, coor, slope_coor)
  
   # on renvoie le nombre de prticules restantes et les coordonnees des 
   # particules
   return [nb_remaining_particles, coor]

# fonction qui depose les particules dans un "cylindre", pour un cisaillement
# de Couette
def depositInCouette2D(radii, rint, rext, deposited_radii=None, deposited_coor=None):
   '''[nb_remaining_particles, coor]=depositInCouetteD(radii, rint, rext, deposited_radii=None, deposited_coor=None):

   this function deposits circular particles in container designed for a Couette shear

   parameters:

   - radii: radii of the particles
   - rint: internal radius of the ring occupied by particles
   - rext: external radius of the ring occupied by particles

   optional parameters:

   - deposited_radii=None: radii of particles supposed to be in the box before the deposit
   - deposited_coor=None: radii of these deposited particles  

   returned values:

   - nb_remaining_particles: number of deposited particles
   - coor: coordinates of the deposited particles [x1, y1, x2, y2, ...]

   WARNING: 

   1. this function changes the radii list since it nullifies radii of the 
      particles out of the box. So, after calling this function, only the 
      first nb_remaining_particles particles are of interest 
   2. to avoid interpenetrations between particles and walls, this function
      uses a shrink based on the size of the particles'''

   # on recupere le nombre de grains
   nb_particles=len(radii)
   # on recupere le rayon du plus gros grain
   radius_max=max(radii)
   # on place une grosse particule pour represente le cylindre interieur
   big_radii=numpy.array([rint])
   big_coor=numpy.array([rext, rext])
   # si on a donne une liste de grosses particules a ajouter au depot
   if (deposited_radii != None and deposited_coor != None):
      # on les ajoute
      big_radii=numpy.concatenate( (big_radii, deposited_radii) )
      big_coor=numpy.concatenate( (big_coor, deposited_coor) )
   # on realise un depot autour de grosses particules, dans une boite de 
   # largueur 2.*rext
   coor=lmgc90.deposit2D_Heterogeneous(radii, 2.*rext, big_radii, big_coor, 2*nb_particles)
   # on definit un contour circulaire, de rayon rext et centre en [rext, rext]
   # N.B.: on definit dans un container legerement plus petit, pour eliminer les
   #       les interpentrations avec le cylindre exterieur
   slope_coor=numpy.zeros(162, 'd')
   for i in range(0, 81, 1):
      slope_coor[2*i]=rext + (rext - radius_max)*math.cos(math.pi*(1. - i*0.025))
      slope_coor[2*i + 1]=rext + (rext - radius_max)*math.sin(math.pi*(1. - i*0.025))

   # on enleve les grains hors du contour
   nb_remaining_particles=lmgc90.cut2D_Cut(radii, coor, slope_coor)
  
   # on renvoie le nombre de prticules restantes et les coordonnees des 
   # particules
   return [nb_remaining_particles, coor]

# fonction qui depose les particules de sorte a remplir un demi-tambour
def depositInDrum2D(radii, r, deposited_radii=None, deposited_coor=None):
   '''[nb_remaining_particles, coor]=depositInDrum2D(radii, deposited_radii=None, deposited_coor=None):

   this function deposits circular particles in the lower half part of a drum

   parameters:

   - radii: radii of the particles
   - r: radius of the container 

   optional parameters:

   - deposited_radii=None: radii of particles supposed to be in the box before the deposit
   - deposited_coor=None: radii of these deposited particles  

   returned values:

   - nb_remaining_particles: number of deposited particles
   - coor: coordinates of the deposited particles [x1, y1, x2, y2, ...]

   WARNING: 

   1. this function changes the radii list since it nullifies radii of the 
      particles out of the box. So, after calling this function, only the 
      first nb_remaining_particles particles are of interest 
   2. to avoid interpenetrations between particles and walls, this function
      uses a shrink based on the size of the particles'''

   # on recupere le nombre de grains
   nb_particles=len(radii)
   # on recupere le rayon du plus gros grain
   radius_max=max(radii)

   # on realise un depot sous gravite, dans une boite de largueur 2.*r

   # si on a donne une liste de particules deja presentes dans la boite
   if (deposited_radii != None and deposited_coor != None):
      # on utilise la methode qui en tient compte
      coor=lmgc90.deposit2D_GravityAndBigParticles(radii, 2.*r,
         deposited_radii, deposited_coor, 2*nb_particles)
   # sinon,
   else:
      # on utilise la methode de base 
      coor=lmgc90.deposit2D_Gravity(radii, 2.*r, 2*nb_particles)

   # on definit le contour a utiliser...
   slope_coor=numpy.zeros(84, 'd')
   # ... en deux parties :
   #   - une ligne horizontale pour remplir a moitie
   slope_coor[0]=radius_max
   slope_coor[1]=r
   #   - un demi-cercle pour decrire le fond du demi-tambour
   # N.B.: on definit dans un container legerement plus petit, pour eliminer les
   #       les interpentrations avec la paroi
   for i in range(0, 41, 1):
      slope_coor[2*(i + 1)]=r + (r - radius_max)*math.cos(math.pi*i*0.025)
      slope_coor[2*(i + 1) + 1]=r - (r - radius_max)*math.sin(math.pi*i*0.025)

   # on enleve les grains hors du contour
   nb_remaining_particles=lmgc90.cut2D_Cut(radii, coor, slope_coor)
  
   # on renvoie le nombre de prticules restantes et les coordonnees des 
   # particules
   return [nb_remaining_particles, coor]
