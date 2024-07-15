
import numpy as np

from . import utils

def inters2polyg( f2f, inters_prprx ):
    """
    Generate the list of polygons from
    the PRPrx interactions array of LMGC90
    (numpy structured array)

    Each polygon is a tuple with coordinates,
    reaction in global frame and normal reaction value
    """

    polygs = []
    idx = 0
    for i_f2f in range(f2f[0]):
      idx += 1

      nb_v = f2f[idx]

      polyg_idx = np.where( np.isin( inters_prprx['icdan'], f2f[idx+1:idx+nb_v+1] ) )[0]

      loc  = inters_prprx[polyg_idx]
      coor = loc['coor']
      rn   = loc['rl'][:,1]

      reac = np.matmul( loc['uc'], loc['rl'][:,:,np.newaxis] )

      polygs.append( (coor, reac[:,:,0], rn) )
      idx += nb_v

    return polygs


def polyg2ck( polygs ):
    """
    Compute the central kernel of each input polygons

    The central kernel is a tuple with coordinates of the kernel,
    the equivalent normal stress on the polygon, coordinates of the center
    of pressure and if this point is inside the central kernel
    """

    
    ck = []

    i_f2f = 0
    for coor, reac, rn in polygs:
        i_f2f += 1

        # getting 3D coordinates projected on the polygon plan
        frame, orig = utils.space_mapping(coor)
        mapped_coor = np.matmul(frame,(coor-orig).T).T

        xc  = utils.pressure_center(mapped_coor, rn)
        s_n = utils.sigma(mapped_coor, rn)

        # in case the polygon is not convex, its shape
        # maybe different from the input...
        ck_coor2d = utils.central_kernel(mapped_coor[:,:2])
        ck_coor   = np.zeros( [ck_coor2d.shape[0],3] )
        ck_coor[:,:2] = ck_coor2d[:,:]

        # check if center of pressure is in central kernel
        is_in = utils.is_inside(xc, ck_coor2d)

        unmapped_ck = np.matmul(frame.T,ck_coor.T).T + orig
        unmapped_xc = np.matmul(frame.T,xc).T + orig

        ck.append( (unmapped_ck, s_n, unmapped_xc, is_in,) )

    return ck


def get( f2f, inters ):

    polygs = inters2polyg( f2f, inters )
    ck     = polyg2ck( polygs )

    return polygs, ck
