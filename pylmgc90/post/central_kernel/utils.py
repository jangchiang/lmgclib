
import math

import numpy as np
from scipy.spatial import ConvexHull

def geometric_center(a):
     """
     compute geometrique center of a polygon
     """

     nb_lig = a.shape[0]
     G  = np.sum(a,axis=0)
     G /= nb_lig

     return G

def centroid(a):
    """
    Compute centroid of a polygon   
    source : http://paulbourke.net/geometry/polygonmesh/
    """

    #last vertices index
    last = a.shape[0]-1
    g = np.zeros( [2] )
    for i in range(-1, last):
        g[:] += (a[i,:]+a[i+1,:])*np.cross(a[i,:], a[i+1,:])

    g[:] /= 6*area(a)

    return g 


def area(a):
    """
    Compute area of a polygon
    source : http://paulbourke.net/geometry/polygonmesh/
    """

    #last vertices index
    last = a.shape[0]-1

    area = 0.
    for i in range(-1, last):
        area += np.cross( a[i,:], a[i+1,:] )

    return area/2.


def moment_inertia(a):
    """
    Compute inertia of polygon along x and y axis, relatively to centroid
    source : http://paulbourke.net/geometry/polygonmesh/
    """

    g = centroid(a)
    polyg = a - g

    #last vertices index
    last = polyg.shape[0]-1

    I = np.zeros([2])
    for i in range(-1, last):
        I[:] += np.cross(polyg[i,:],polyg[i+1,:]) * (polyg[i,:]**2+polyg[i,:]*polyg[i+1,:]+polyg[i+1,:]**2)

    I[:] /= 12.
    I[0], I[1] = I[1], I[0]

    return I


def inertial_ellipse(a, t=None):

    t  = np.linspace(0., 2*np.pi, 101) if t is None else t
    S  = area(a)
    I  = moment_inertia(a)
    g  = centroid(a)
    ellipse = np.empty([t.size,2])

    ellipse[:,0]=math.sqrt(I[1]/S)
    ellipse[:,1]=math.sqrt(I[0]/S)
    ellipse[:,0] *= np.cos(t)
    ellipse[:,1] *= np.sin(t)
    ellipse[:,:] += g

    return ellipse

def antipolars(a, I=None):
    """
    Compute the antipolars from the vertices of a polygon
    Uses the convex hull and antipolars are provided as
    slope and intercept.
    Notes : an np.inf value for slope means a y-axis parallel line
    """

    g = centroid(a)
    polyg = a - g
    S  = area(a)
    I  = moment_inertia(a) if I is None else I

    hull = ConvexHull(polyg)
    nb_l = hull.vertices.shape[0]

    h=np.zeros([nb_l,2]) 
    for i, j in enumerate(hull.vertices) :
        if hull.points[j,0] == 0. :
            h[i,0]  =0
            h[i,1] +=-I[0]/(S*hull.points[j,1])#+g[1] #ordonnée à l'origine
        elif hull.points[j,1] == 0. :
            h[i,0]  = np.inf
            h[i,1] +=-I[1]/(S*hull.points[j,0])#abscisse à l'origine
        else:
            h[i,0] +=-I[0]*hull.points[j,0]/(I[1]*hull.points[j,1])
            h[i,1] +=-I[0]/(S*hull.points[j,1])#+g[1]
    return h


def central_kernel(a):

    h = antipolars(a)
    nb_l = h.shape[0]

    inter= np.zeros([nb_l,2])
    for i in range(-1, nb_l-1):

        # sanity check
        assert h[i,0] != h[i+1,0], 'looking for intersection of parallel lines'

        # pathologic cases of line along y axis
        if h[i,0] == np.inf:
            inter[i,0] = h[i,1]
            inter[i,1] = h[i+1,0]*inter[i,0]+h[i+1,1]
        elif h[i+1,0] == np.inf:
            inter[i,0] = h[i+1,1]
            inter[i,1] = h[i,0]*inter[i,0]+h[i,1]
        else:
            inter[i,0] = (h[i+1,1]-h[i,1])/(h[i,0]-h[i+1,0])
            inter[i,1] = h[i,0]*inter[i,0]+h[i,1]
    return inter
    

def space_mapping(a):
    """
    Compute the space mapping (translation and rotation)
    allowing to compute the polygon in its own frame

    The mapping is then computed with:
      new = np.matmul( frame, (old-offset).T )
    The reverse mapping is then computed with:
      old = np.matmul( frame.T, new ).T + offset
    """

    f1= a[1,:]-a[0,:]
    f1=f1/np.linalg.norm(f1)

    i = 2
    while i < a.shape[0] :
        v  = a[i,:]-a[0,:]
        f3 = np.cross(f1,v)
        n3 = np.linalg.norm(f3)
        if n3 != 0.:
          break
        i=i+1

    if i == a.shape[0]:
      print('[space_mapping:error] all points aligned')
      raise ValueError

    f3 = f3/n3
    f2 = np.cross(f3,f1)

    frame  = np.array( [f1,f2,f3], dtype=float)
    offset = geometric_center( a )

    return frame, offset

  
def nc_each_rectangle(f2f):
    """
    Compute the central kernel of each surface of f2f using the fonction central_kernel
    """
    final_result = []
    i_f2f = 0
    for coor, reac, rn in f2f:
        i_f2f += 1

        f, g = space_mapping(coor)
        new_coor = np.matmul(f,(coor-g).T).T

        s_n   = sigma(new_coor, rn)

        last_l = np.zeros(new_coor.shape)
        last_l[:,:2] = central_kernel(new_coor[:,:2])
        xc    = center_pression(new_coor, rn)

        is_in = insidePolygon(last_l[:,:2], xc)

        new_last = np.matmul(f.T,last_l.T).T + g
        new_xc   = np.matmul(f.T,xc).T + g
        final_result.append( (new_last, s_n, new_xc, is_in,) )
    return final_result 

def pressure_center(coor, rn):
    """
    Compute the pressure center of the input polygon with associated normal reaction
    """
    s = np.sum(rn)
    return np.sum(coor*rn[:,np.newaxis],axis=0) / s if s != 0. else geometric_center(coor)


def sigma(coor, rn):  
    """
    Compute the equivalent normal stress of each force applied on each surface of f2f
    """
    a = area(coor[:,:2]) 
    return np.sum(rn) / a if a != 0. else 0.

def is_inside(p,polyg):
    """
    Verifies that the input point is inside the provided polygon
    source : http://paulbourke.net/geometry/polygonmesh/
    """
    nb_polyg = len(polyg)
    compteur=0
    p1=polyg[0]
    for i in range (1,nb_polyg+1):
        p2=polyg[i % (nb_polyg)]
        if (p[1]  > min(p1[1],p2[1])   and
           (p[1] <= max (p1[1],p2[1])) and
           (p[0] <= max(p1[0],p2[0]))  and
           (p1[1] != p2[1]) ):
            xinters = (p[1]-p1[1])*(p2[0]-p1[0])/(p2[1]-p1[1])+p1[0]
            if (p1[0] == p2[0]) or (p[0] <= xinters):
                compteur += 1
        p1=p2
        
    if (compteur % 2 == 0):
        return False
    else:
        return True    


