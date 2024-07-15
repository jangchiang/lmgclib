
import sys, os
import itertools

import collections, fileinput, re

import math
import numpy as np

from . import config
from .lmgc90 import *

if config.is_vtk_display :

  import vtk
  from vtk.util import numpy_support as ns


# wedge element numbering differs beween lmgc and vtk
wedgeMap = list(range(15))
wedgeMap[1] = 2; wedgeMap[2]  = 1
wedgeMap[4] = 5; wedgeMap[5]  = 4
wedgeMap[6] = 8; wedgeMap[8]  = 6
wedgeMap[9] =11; wedgeMap[11] = 9
wedgeMap[13]=14; wedgeMap[14] =13

def init_vtk() :

  if ( not config.is_vtk_display ):
    print( '[WARNING:init] Since vtk module is not available this function does nothing' )
    return

 
########################################
# utilitaires

def buildSegment(coorx,coory,coorz,nx,ny,nz,lr):

   vertex = np.zeros((2,3),dtype=float)
   vertex[0,0]= coorx - (nx*lr)
   vertex[0,1]= coory - (ny*lr)
   vertex[0,2]= coorz - (nz*lr)
   vertex[1,0]= coorx + (nx*lr)
   vertex[1,1]= coory + (ny*lr)
   vertex[1,2]= coorz + (nz*lr)

   return vertex
    

if config.vtkVersion >= (9,0):

  def thresholdVisible(vtk_object, on_cells=False):
      """
      Generate a threshold vtk ojbect to filter only visible
      """

      threshold = vtk.vtkThreshold()
      threshold.SetInputDataObject(0, vtk_object)

      if on_cells:
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS , "Visible");
      else:
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Visible");

      threshold.SetUpperThreshold(1)
      threshold.SetThresholdFunction(threshold.THRESHOLD_UPPER)
      threshold.Update()

      new_object = threshold.GetOutputDataObject(0)
      new_object.GetPointData().RemoveArray("Visible")

      return new_object

else:

  def thresholdVisible(vtk_object, on_cells=False):
      """
      Generate a threshold vtk ojbect to filter only visible
      """

      threshold = vtk.vtkThreshold()
      threshold.SetInputData(vtk_object)

      if on_cells:
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS , "Visible");
      else:
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Visible");

      threshold.ThresholdByUpper(1)
      threshold.Update()

      new_object = threshold.GetOutput()
      new_object.GetPointData().RemoveArray("Visible")

      return new_object

##########################################################
# tactors

class TactorGenerator():
  def __init__(self,name):

    if name == 'CLxxx' or name == 'ALpxx':
      print(name,' contactor not supported yet')
      sys.exit(1)

    self.name=name
    self.GetNb=eval(name+'_GetNb'+name)  
    if name in ['SPHER', 'POLYR', 'PLANx', 'CYLND', 'DNLYC', 'PT3Dx']:
      self.GetTactor2Body=eval(name+'_GetPtr'+name+'2BDYTY')
      self.GetConnectivities=eval(name+'_GetPtrAllConnectivities')
    else:
      self.GetTactor2Body=eval(name+'_GetPtr'+name+'2BDYTY')
      self.GetConnectivities=None

    self.IsVisible=eval(name+'_IsVisible')

    self.GetNbPointOutlines=eval(name+'_GetNbPointOutlines')
    self.GetNbScalarFields=eval(name+'_GetNbScalarFields')
    self.InitOutlines=eval(name+'_InitOutlines')
    self.InitScalarFields=eval(name+'_InitScalarFields')
    self.UpdatePostdata=eval(name+'_UpdatePostdata') 
    
def InitTactor( tacts_dict, name ):

   tact=TactorGenerator(name)

   vals=[]
   # 0 le nombre de tactors
   vals.append(tact.GetNb())
   # 1 la map vers rbdy2
   vals.append(tact.GetTactor2Body())
   # old vals.append(tact.GetTactor2RBDY2(vals[0]))
   # 2 pointeur sur le tableau de contour
   vals.append(tact.InitOutlines)
   # 3 le nombre de pt sur le contour des diskx
   vals.append(tact.GetNbPointOutlines)
   # 4 pointeur sur le tableau de scalar fields
   vals.append(tact.InitScalarFields())
   # 5 le nombre de fields
   vals.append(tact.GetNbScalarFields())
   # 6 update postdata
   vals.append(tact.UpdatePostdata)

   # 7 geometries (as a ugrid)
   outline            = tact.InitOutlines()
   vals[2] = outline
   nb_points_outlines = tact.GetNbPointOutlines()
   ugrid              = vtk.vtkUnstructuredGrid()
   ugrid.Allocate(tact.GetNb())

   # contactors 3D
   if name in ['CYLND', 'DNLYC', 'PLANx', 'POLYR', 'PT3Dx', 'SPHER'] and tact.GetNb() > 0:
      idx    = 0
      connec = tact.GetConnectivities()
      for itact in range(tact.GetNb()):
         # on initialise une liste qui va contenir la connectivite du contacteur
         # le 1er element est le nombre de faces
         faces = vtk.vtkIdList()
         faces.InsertNextId( connec[idx] )
         # on parcourt les faces du tact
         for j in range(connec[idx]):
            # on definit la face
            # le 1er element est le nombre de vertex
            faces.InsertNextId( connec[idx+1] )
            # puis on ajoute les ids des vertex
            for k in range(connec[idx+1]):
               faces.InsertNextId( nb_points_outlines[itact] + connec[idx+2+k]-1 )
            idx += connec[idx+1]+1
         idx += 1
         # on stocke
         ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, faces)

      listOfFields = { "Disp"    : 3,
                       "Velocy"  : 3,
                       "Spin"    : 3,
                       "Reac"    : 3,
                       "Torque"  : 3,
                     }
   # contactors 2D
   else:
      for itact in range(tact.GetNb()):
         # on initialise une liste qui va contenir le contour du contacteur
         edges = vtk.vtkIdList()
         edges.SetNumberOfIds(nb_points_outlines[itact+1]-nb_points_outlines[itact])
         # puis on ajoute les ids des vertex
         for k in range(nb_points_outlines[itact+1]-nb_points_outlines[itact]):
            edges.SetId(k, nb_points_outlines[itact]+k)
         # on stocke
         ugrid.InsertNextCell(vtk.VTK_POLYGON, edges)

      listOfFields = { "Disp"    : 3,
                       "Rot Z"   : 1,
                       "Velocy"  : 3,
                       "Spin Z"  : 1,
                       "Reac"    : 3,
                       "Torque Z": 1,
                     }

   if vals[0] > 0 :
      points = vtk.vtkPoints()
      points.SetNumberOfPoints( nb_points_outlines[vals[0]] )
      ugrid.SetPoints(points)


   cellData = ugrid.GetCellData()
   
   # definition of integer fields
   for field in ["Ids", "Visible", "Material"]:
     f = vtk.vtkIntArray()
     f.SetNumberOfComponents(1)
     f.SetNumberOfTuples(vals[0])
     f.SetName(field)
     cellData.AddArray(f)

   ids = ns.vtk_to_numpy( cellData.GetArray("Ids")      )
   mat = ns.vtk_to_numpy( cellData.GetArray("Material") )
   
   for itact in range(tact.GetNb()):

       ibdyty = vals[1][itact][0]
       ids[itact] = ibdyty
       # so ugly...
       if vals[1][itact][2] == 1:
         mat[itact] = RBDY2_GetBulkBehavNumber( int( ibdyty ) )
       elif vals[1][itact][2] == 2:
         mat[itact] = RBDY3_GetBulkBehavNumber( int( ibdyty ) )
       else:
         mat[itact] = -1


   # definition of real fields
   for field, size in listOfFields.items():
     f = vtk.vtkFloatArray()
     f.SetNumberOfComponents(size)
     f.SetNumberOfTuples(vals[0])
     f.SetName(field)
     cellData.AddArray(f)

   vals.append( ugrid )

   # 8 is body visible
   vals.append(tact.IsVisible)
   #
   tacts_dict[ name ] = vals 

def InitRigidsToVTK(dim):

   if dim == 2 :
     nbm = RBDY2_GetNbRBDY2()
   else:
     nbm = RBDY3_GetNbRBDY3()

   ug = vtk.vtkUnstructuredGrid()

   points = vtk.vtkPoints()
   points.SetNumberOfPoints(nbm)
   ug.SetPoints(points)

   cells = vtk.vtkCellArray()
   off = np.fromiter( range(nbm+1), dtype=int, count=nbm+1 )

   if config.vtkVersion < (9,0) :
       vconnec = vtk.vtkIdTypeArray()
       vconnec.SetNumberOfValues(2*nbm)
       connec  = ns.vtk_to_numpy(vconnec)
       connec[::2]  = 1
       connec[1::2] = off[:nbm]
       cells.SetCells(nbm,vconnec)
   else:
       voffset = vtk.vtkIdTypeArray()
       voffset.SetNumberOfValues(nbm+1)
       offset  = ns.vtk_to_numpy(voffset)

       vconnec = vtk.vtkIdTypeArray()
       vconnec.SetNumberOfValues(nbm)
       connec  = ns.vtk_to_numpy(vconnec)

       offset[:] = off[:]
       connec[:] = offset[:nbm]

       cells.SetData(voffset, vconnec)

   ug.SetCells(vtk.VTK_VERTEX, cells)

   pointData = ug.GetPointData()

   # definition of integer fields
   for field in ["Ids", "Visible"]:
     f = vtk.vtkIntArray()
     f.SetNumberOfComponents(1)
     f.SetNumberOfTuples(nbm)
     f.SetName(field)
     pointData.AddArray(f)


   listOfFields = { "Disp"   : 3, 
                    "Velocy" : 3,
                    "Fext"   : 3,
                    "Reac"   : 3
                  }

   if dim==2 :
     listOfFields["Rot Z"]    = 1
     listOfFields["Spin Z"]   = 1
     listOfFields["Mext Z"]   = 1
     listOfFields["Torque Z"] = 1
   else:
     listOfFields["Spin"]   = 3
     listOfFields["Mext"]   = 3
     listOfFields["Torque"] = 3
     listOfFields["alpha"]  = 3
     listOfFields["beta"]   = 3
     listOfFields["gamma"]  = 3

   # definition of real fields
   for field, size in listOfFields.items():
     f = vtk.vtkFloatArray()
     f.SetNumberOfComponents(size)
     f.SetNumberOfTuples(nbm)
     f.SetName(field)
     pointData.AddArray(f)

   # setting Ids only once :
   ids = ns.vtk_to_numpy( pointData.GetArray("Ids") )
   ids[:] = off[1:]

   return ug 


def pushTactors2D(tacts_dict,Append,kw):

   init_vtk()

   nb_write = 0

   #if len(kw) != 0:
   #  print 'user fields'


   for key in tacts_dict:
      #print 'managing: ',key

      tact=tacts_dict[key]

      # on recupere le pointeur sur outlines
      outlines=tact[2]
      
      if np.size(outlines) == 0:
        continue

      # update postdata
      # calcul le outlines
      tact[6]()
      
      ugrid = tact[7]
      
      # update contactor vertex position
      points = ugrid.GetPoints()
      pcoors = ns.vtk_to_numpy( points.GetData() )

      pcoors[:,:2] = outlines[:,:]
      pcoors[:, 2] = 0.

      # update contactor field value
      cellData = ugrid.GetCellData()
      Vis = ns.vtk_to_numpy( cellData.GetArray("Visible")  )
      Dx  = ns.vtk_to_numpy( cellData.GetArray("Disp")     )
      Rz  = ns.vtk_to_numpy( cellData.GetArray("Rot Z")    )
      Vx  = ns.vtk_to_numpy( cellData.GetArray("Velocy")   )
      Wz  = ns.vtk_to_numpy( cellData.GetArray("Spin Z")   )
      Fx  = ns.vtk_to_numpy( cellData.GetArray("Reac")     )
      Mz  = ns.vtk_to_numpy( cellData.GetArray("Torque Z") )

      field  = tact[4][:,:]

      Dx[:,:2] = field[:, 0:2]
      Dx[:, 2] = 0.
      Rz[:]    = field[:,  2 ]
      Vx[:,:2] = field[:, 3:5]
      Vx[:, 2] = 0.
      Wz[:]    = field[:,  5 ]
      Fx[:,:2] = field[:, 6:8]
      Mz[:]    = field[:,  8 ]

      for itact in range(tact[0]):

         if tact[8](int(itact+1)):
            Vis[itact]= 1
         else:
            Vis[itact]= 0

      # add user field
      if len(kw) != 0:
         for name, dico in list(kw.items()):
           #print '-> ',name
            if not key in dico[0].keys():
                print(key,' not present in dictionary related to',name,' user field')
                raise Exception
            val  = dico[0][key]
            addUserField(name, val, cellData, 2)

      # threshold sur la visibilite
      tgrid = thresholdVisible(ugrid, on_cells=True)
      nb_write += tgrid.GetNumberOfCells()

      Append.AddInputData(tgrid)
      #Append.AddInputData(ugrid)

   return nb_write

def pushTactors3D(tacts_dict,Append,kw):

   init_vtk()

   nb_write = 0

   #if len(kw) != 0:
   #  print 'user fields'

   for key in tacts_dict:
      #print('managing:', key)

      tact=tacts_dict[key]

      # on recupere le pointeur sur outlines
      outlines=tact[2]

      if np.size(outlines) == 0:
        continue

      # update postdata
      # calcul le outlines
      tact[6]()

      ugrid = tact[7]
      
      # update contactor vertex position
      points = ugrid.GetPoints()
      pcoors = ns.vtk_to_numpy( points.GetData() )
      pcoors[:,:] = outlines[:,:]

      # update contactor field value
      cellData = ugrid.GetCellData()
      Vis = ns.vtk_to_numpy( cellData.GetArray("Visible") )
      Dx  = ns.vtk_to_numpy( cellData.GetArray("Disp")    )
      Vx  = ns.vtk_to_numpy( cellData.GetArray("Velocy")  )
      Wz  = ns.vtk_to_numpy( cellData.GetArray("Spin")    )
      Fx  = ns.vtk_to_numpy( cellData.GetArray("Reac")    )
      Mz  = ns.vtk_to_numpy( cellData.GetArray("Torque")  )

      field = tact[4][:,:]

      Dx[:,:] = field[:, 0: 3]
      Vx[:,:] = field[:, 3: 6]
      Fx[:,:] = field[:, 9:12]

      for itact in range(tact[0]):
         #print('itact = ', itact+1)

         ibdyty = tact[1][itact][0]

         if tact[8](int(itact+1)):
            Vis[itact] = 1
         else:
            Vis[itact] = 0

         frame = RBDY3_GetBodyMatrix('IF___',int(ibdyty))

         tmp = frame.T.dot(field[itact,6:9])
         Wz[itact,:] = tmp
         tmp = frame.T.dot(field[itact,12:15])
         Mz[itact,:] = tmp

      # add user field
      #assert(val.shape[0] == tact[0]), "wrong number of input in user field"
      if len(kw) != 0:
         for name, dico in list(kw.items()):
            if not key in dico[0].keys():
                print(key,' not present in dictionary related to',name,' user field')
                raise Exception
            val  = dico[0][key]
            addUserField(name, val, cellData, 3)

      tgrid = thresholdVisible(ugrid, on_cells=True)
      nb_write += tgrid.GetNumberOfCells()

      Append.AddInputData(tgrid)
      
   return nb_write

def InitTactorsToVTK(tacts_list,tacts_dict):
   for tact in tacts_list:
      InitTactor(tacts_dict,tact)

#######################################
# inter

def pushInters(dim,inters,kw):

   init_vtk()

   nb_inter = inters.size
   off = np.fromiter( range(nb_inter+1), dtype=int, count=nb_inter+1 )

   reac = np.matmul( inters['uc'], inters['rl'][:,:,np.newaxis] )
   reac = reac[:,:,0]

   # field name if integer and number of tuples
   # using inters when it is not defined... and in lambda... so ugly !!!
   fielddef = [ (    'id',  True, 1, off[:nb_inter]          ,),
                ( 'inter',  True, 1, inters[ 'inter'][:]     ,),
                ( 'icdan',  True, 1, inters[ 'icdan'][:]     ,),
                ( 'cdbdy',  True, 1, inters[ 'cdbdy'][:]     ,),
                ( 'anbdy',  True, 1, inters[ 'anbdy'][:]     ,),
                ('icdbdy',  True, 1, inters['icdbdy'][:]     ,),
                ('ianbdy',  True, 1, inters['ianbdy'][:]     ,),
                ( 'behav',  True, 1, inters[ 'behav'][:]     ,),
                ('status',  True, 1, inters['status'][:]     ,),
                (     'T', False, 3, inters[    'uc'][:][:,0],),
                (     'N', False, 3, inters[    'uc'][:][:,1],),
                (     'R', False, 3, reac[:,:]                ,),
                (   'rlt', False, 1, inters[    'rl'][:,0]   ,),
                (   'rln', False, 1, inters[    'rl'][:,1]   ,),
                (   'vlt', False, 1, inters[    'vl'][:,0]   ,),
                (   'vln', False, 1, inters[    'vl'][:,1]   ,),
                (   'gap', False, 1, inters[    'gapTT'][:]  ,),
              ]

   if dim == 3:
     fielddef.extend( [ (  'S', False, 3, inters['uc'][:][:,2],),
                        ('rls', False, 1, inters['rl'][:,2]   ,),
                        ('vls', False, 1, inters['vl'][:,2]   ,),
                      ] )

   #print('nb inter ',nb_inter)

   if len(kw) != 0:
     for name, efield in list(kw.items()):
       #print('-> ',name)
       if not efield.shape[0] == nb_inter :
         print(name,' user field is of size ', efield.shape[0], ' but should be ', nb_inter)
         raise Exception

   # Allocate poydata, points, cells and vertices
   pdata = vtk.vtkPolyData()

   points = vtk.vtkPoints()
   points.SetNumberOfPoints(nb_inter)
   pcoors = ns.vtk_to_numpy( points.GetData() )

   cells = vtk.vtkCellArray()
   if config.vtkVersion < (9,0) :
       vconnec = vtk.vtkIdTypeArray()
       vconnec.SetNumberOfValues(2*nb_inter)
       connec  = ns.vtk_to_numpy(vconnec)
       connec[::2]  = 1
       connec[1::2] = off[:nb_inter]
       cells.SetCells(nb_inter,vconnec)
   else:
       voffset = vtk.vtkIdTypeArray()
       voffset.SetNumberOfValues(nb_inter+1)
       offset  = ns.vtk_to_numpy(voffset)
  
       vconnec = vtk.vtkIdTypeArray()
       vconnec.SetNumberOfValues(nb_inter)
       connec  = ns.vtk_to_numpy(vconnec)
  
       offset[:] = off[:]
       connec[:] = offset[:-1]

       cells.SetData(voffset, vconnec)

   pdata.SetPoints( points )
   pdata.SetVerts( cells )

   #pointData = pdata.GetPointData()
   cellData  = pdata.GetCellData()

   # defining fields
   field_list = []
   for field in fielddef:
     name, is_int, size, _ = field
     new_field = vtk.vtkIntArray() if is_int else vtk.vtkFloatArray()
     new_field.SetName(name)
     new_field.SetNumberOfComponents(size)
     new_field.SetNumberOfTuples(nb_inter)
     #pointData.AddArray(new_field)
     cellData.AddArray(new_field)

     field_list.append( ns.vtk_to_numpy(new_field) )

   pcoors[:,:dim] = inters['coor']
   if dim == 2:
     pcoors[:,2] = 0.

   #uc = np.transpose( inters['uc'], axes=[0,2,1] )
   #rl = inters['rl'][:,:,np.newaxis]

   # setting inter type field
   for field, fdef in zip(field_list, fielddef):
     _, _, size, access = fdef
     if size == 1 :
       field[:] = access[:]
     else:
       if dim == 3:
         field[:,:size] = access[:,:size]
       else:
         field[:,:2] = access[:,:]
         field[:, 2] = 0.

   # on pousse les champs user
   if len(kw) != 0:
      for name, efield in list(kw.items()):
         addUserField(name, efield, cellData, dim)

   return pdata


def writeIntersToVTK(fichier,fid,inters,dim,**kw):

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writeIntersToVTK] Since vtk module is not available this function does nothing' )
      return

   # fichiers ptc.vtu
   File_ptc = vtk.vtkXMLPolyDataWriter()
   File_ptc.SetFileName(fichier)
   # rm: what is the real use of this ?
   #File_ptc.SetIdTypeToInt32()
   #File_ptc.SetIdTypeToInt64()
   File_ptc.SetDataMode(vtk.vtkXMLWriter.Binary)

   pdata = pushInters(dim,inters,kw)

   time=TimeEvolution_GetTime()
   impr ='<DataSet timestep="%s" group="" part="0" file="%s"/>\n' % (time,'./'+os.path.basename(fichier))
   impr+= '</Collection>\n</VTKFile>'

   # remove closing part of pvd
   removeLines(fid)
   # add new lines of pvd
   with open(fid,'a') as f:
     f.write(impr)

   File_ptc.SetInputData(pdata)
   File_ptc.Write()


def pushF2f(inters,f2f_inters,Append_f2f,kw):

   init_vtk()

   nb_f2f = f2f_inters[0]

   if len(kw) != 0:
     for name, efield in list(kw.items()):
       #print('-> ',name)
       if not efield.shape[0] == nb_inter :
         print(name,' user field is of size ', efield.shape[0], ' but should be ', nb_inter)
         raise Exception

   pdata = vtk.vtkPolyData()
   pdata.Allocate(nb_f2f)

   points = vtk.vtkPoints()
   points.SetNumberOfPoints(inters.size+nb_f2f)

   cells = vtk.vtkCellArray()
   #verts = vtk.vtkCellArray()

   cellData  = pdata.GetCellData()
   pointData = pdata.GetPointData()

   for i_field in ['ids', 'icdbdy', 'icdtac', 'ianbdy', 'iantac',]:
     i_cf = vtk.vtkIntArray()
     i_cf.SetNumberOfComponents(1)
     i_cf.SetNumberOfTuples(nb_f2f)
     i_cf.SetName(i_field)
     cellData.AddArray(i_cf)
   for r_field in ['ST','SN','SS','area']:
     r_cf = vtk.vtkDoubleArray()
     r_cf.SetNumberOfComponents(1)
     r_cf.SetNumberOfTuples(nb_f2f)
     r_cf.SetName(r_field)
     cellData.AddArray(r_cf)

   ids   = cellData.GetArray("ids")
   cdbdy = cellData.GetArray("icdbdy")
   anbdy = cellData.GetArray("ianbdy")
   cdtac = cellData.GetArray("icdtac")
   antac = cellData.GetArray("iantac")
   ST    = cellData.GetArray("ST")
   SN    = cellData.GetArray("SN")
   SS    = cellData.GetArray("SS")
   area  = cellData.GetArray("area")

   r_fields = [('rln',1), ('rlt',1), ('rls',1), ('Reac',3), ('T',3), ('N',3), ('S',3),]
   for r_field, r_size in r_fields:
     r_pf = vtk.vtkDoubleArray()
     r_pf.SetNumberOfComponents(r_size)
     r_pf.SetNumberOfTuples(inters.size+nb_f2f)
     r_pf.SetName(r_field)
     pointData.AddArray(r_pf)
   cop  = vtk.vtkIntArray()
   cop .SetNumberOfComponents(1)
   cop .SetNumberOfTuples(inters.size+nb_f2f)
   cop .SetName('Center')
   pointData.AddArray(cop)

   rlt  = pointData.GetArray("rlt")
   rln  = pointData.GetArray("rln")
   rls  = pointData.GetArray("rls")
   Reac = pointData.GetArray("Reac")
   T    = pointData.GetArray("T")
   N    = pointData.GetArray("N")
   S    = pointData.GetArray("S")

   reac = np.matmul( inters['uc'], inters['rl'][:,:,np.newaxis] )
   reac = reac[:,:,0]

   i2p = {}
   p_idx = 0
   idx = 1
   for i_f2f in range(nb_f2f):
     nb_inters = f2f_inters[idx]
     idx += 1

     # map the f2f points to a polygon
     polyg = vtk.vtkPolygon()
     polyg.GetPointIds().SetNumberOfIds(nb_inters)

     surf  = 0.
     SNval = 0.
     STval = 0.
     SSval = 0.

     cpres = 0.
     ccoor = 0.
     creac = 0.

     i_prev = f2f_inters[idx+nb_inters-1]-1
     for i_inter in range(nb_inters):
         i_point = f2f_inters[idx+i_inter]-1

         STval  += inters[i_point]['rl'][0]
         SNval  += inters[i_point]['rl'][1]
         SSval  += inters[i_point]['rl'][2]
         surf   +=  np.cross( inters[i_prev]['coor'], inters[i_point]['coor'] )

         ccoor += inters[i_point]['coor']
         cpres += inters[i_point]['rl'][1] * inters[i_point]['coor']
         creac += reac[i_point]

         i_prev  = i_point
         if i_point not in i2p.keys():
           i2p[i_point] = p_idx

           points.SetPoint(p_idx, *inters[i_point]['coor'])

           cop.SetTuple1(  p_idx,  0)
           rlt.SetTuple1(  p_idx,  inters[i_point]['rl'][0])
           rln.SetTuple1(  p_idx,  inters[i_point]['rl'][1])
           rls.SetTuple1(  p_idx,  inters[i_point]['rl'][2])
           T.SetTuple3(    p_idx, *inters[i_point]['uc'][:,0])
           N.SetTuple3(    p_idx, *inters[i_point]['uc'][:,1])
           S.SetTuple3(    p_idx, *inters[i_point]['uc'][:,2])
           Reac.SetTuple3( p_idx, *reac[i_point])

           p_idx += 1

         polyg.GetPointIds().SetId(i_inter, i2p[i_point])

     cells.InsertNextCell(polyg)
     idx += nb_inters

     ids.SetTuple1(i_f2f, i_f2f)
     cdbdy.SetTuple1( i_f2f, inters[i_point]['icdbdy'] )
     anbdy.SetTuple1( i_f2f, inters[i_point]['ianbdy'] )
     cdtac.SetTuple1( i_f2f, inters[i_point]['icdtac'] )
     antac.SetTuple1( i_f2f, inters[i_point]['iantac'] )

     cpres = cpres / SNval if SNval != 0. else ccoor / nb_inters
     points.SetPoint(p_idx, *cpres)
     cop.SetTuple1(  p_idx,  1)
     rlt.SetTuple1(  p_idx,  STval)
     rln.SetTuple1(  p_idx,  SNval)
     rls.SetTuple1(  p_idx,  SSval)
     T.SetTuple3(    p_idx, *inters[i_point]['uc'][:,0])
     N.SetTuple3(    p_idx, *inters[i_point]['uc'][:,1])
     S.SetTuple3(    p_idx, *inters[i_point]['uc'][:,2])
     Reac.SetTuple3( p_idx, *creac)
     #vertex = vtk.vtkVertex()
     #vertex.GetPointIds().SetId(0, p_idx)
     #verts.InsertNextCell(vertex)
     p_idx += 1

     surf  = np.linalg.norm( surf )
     SNval /= surf
     STval /= surf
     SSval /= surf

     area.SetTuple1( i_f2f, surf)
     SN.SetTuple1( i_f2f, SNval)
     ST.SetTuple1( i_f2f, STval)
     SS.SetTuple1( i_f2f, SSval)

     # on pousse les champs user
     if len(kw) != 0:
        for name, efield in list(kw.items()):
           addUserField(name, efield, cellData, dim)

   pdata.SetPoints(points)
   pdata.SetPolys(cells)
   #pdata.SetVerts(verts)

   Append_f2f.AddInputData(pdata)

   return 1


def writeF2fToVTK(fichier, f2f, inters, f2f_inters, **kw):

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writeF2fToVTK] Since vtk module is not available this function does nothing' )
      return

   # fichiers ptc.vtu
   File_f2f = vtk.vtkXMLPolyDataWriter()
   File_f2f.SetFileName(fichier)
   File_f2f.SetDataMode(vtk.vtkXMLWriter.Binary)

   Append_f2f = vtk.vtkAppendPolyData()

   nb = pushF2f(inters, f2f_inters, Append_f2f, kw)

   time=TimeEvolution_GetTime()
   impr ='<DataSet timestep="%s" group="" part="0" file="%s"/>\n' % (time,'./'+os.path.basename(fichier))
   impr+= '</Collection>\n</VTKFile>'

   # remove closing part of pvd
   removeLines(f2f)
   # add new lines of pvd
   with open(f2f,'a') as f:
     f.write(impr)

   File_f2f.SetInputConnection(Append_f2f.GetOutputPort())
   File_f2f.Write()


############################
# rigids

def pushRigid2D(ug,kw):

   init_vtk()

   # print('managing: rigid')
   nbm = RBDY2_GetNbRBDY2()

   # rigids vertex position
   points = ug.GetPoints()
   pcoors = ns.vtk_to_numpy(points.GetData())
   pcoors[:,2] = 0.

   # rigids field value
   pointData = ug.GetPointData()
   Vis = ns.vtk_to_numpy( pointData.GetArray("Visible") )
   Dx  = ns.vtk_to_numpy( pointData.GetArray("Disp")    )
   Vx  = ns.vtk_to_numpy( pointData.GetArray("Velocy")  )
   Fx  = ns.vtk_to_numpy( pointData.GetArray("Fext")    )
   Rx  = ns.vtk_to_numpy( pointData.GetArray("Reac")    )

   Dz = ns.vtk_to_numpy( pointData.GetArray("Rot Z")    )
   Wz = ns.vtk_to_numpy( pointData.GetArray("Spin Z")   )
   Mz = ns.vtk_to_numpy( pointData.GetArray("Mext Z")   )
   Rz = ns.vtk_to_numpy( pointData.GetArray("Torque Z") )

   Dx[:,2] = 0.
   Vx[:,2] = 0.
   Fx[:,2] = 0.
   Rx[:,2] = 0.

   for i in range(nbm):
     # print('rigid : %i'%(i+1))

     if RBDY2_IsVisible(i+1):
        Vis[i] = 1
     else:
        Vis[i] = 0

     coor = RBDY2_GetBodyVector('Coor_',i+1)
     pcoors[i,:2] = coor[:2]

     field = RBDY2_GetBodyVector('X____',i+1)
     Dx[i,:2] = field[:2]
     Dz[i]    = field[ 2]

     field = RBDY2_GetBodyVector('V____',i+1)
     Vx[i,:2] = field[:2]
     Wz[i]    = field[ 2]

     field = RBDY2_GetBodyVector('Fext_',i+1)
     Fx[i,:2] = field[:2]
     Mz[i]    = field[ 2]

     field = RBDY2_GetBodyVector('Reac_',i+1)
     Rx[i,:2] = field[:2]
     Rz[i]    = field[ 2]

   # user fields
   # check size of input array ?
   if len(kw) != 0:
     for name,value in list(kw.items()):
       addUserField(name, value, pointData, 2)

   # threshold on visibility
   tg  = thresholdVisible(ug)

   return tg


def pushRigid3D(ug,kw):

   init_vtk()

   # print('managing: rigid')
   nbm = RBDY3_GetNbRBDY3()

   # rigids vertex position
   points = ug.GetPoints()
   pcoors = ns.vtk_to_numpy(points.GetData())

   # rigids field value
   pointData = ug.GetPointData()
   Vis = ns.vtk_to_numpy( pointData.GetArray("Visible") )
   Dx  = ns.vtk_to_numpy( pointData.GetArray("Disp")    )
   Vx  = ns.vtk_to_numpy( pointData.GetArray("Velocy")  )
   Fx  = ns.vtk_to_numpy( pointData.GetArray("Fext")    )
   Rx  = ns.vtk_to_numpy( pointData.GetArray("Reac")    )

   Wz    = ns.vtk_to_numpy( pointData.GetArray("Spin")   )
   Mz    = ns.vtk_to_numpy( pointData.GetArray("Mext")   )
   Rz    = ns.vtk_to_numpy( pointData.GetArray("Torque") )
   alpha = ns.vtk_to_numpy( pointData.GetArray("alpha")  )
   beta  = ns.vtk_to_numpy( pointData.GetArray("beta")   )
   gamma = ns.vtk_to_numpy( pointData.GetArray("gamma")  )


   for i in range(nbm):
     # print('rigid : %i'%(i+1))

     if RBDY3_IsVisible(i+1):
        Vis[i] = 1
     else:
        Vis[i] = 0

     coor = RBDY3_GetBodyVector('Coor_',i+1)
     pcoors[i,:3] = coor[:3]

     frame = RBDY3_GetBodyMatrix('IF___',i+1)

     field = RBDY3_GetBodyVector('X____',i+1)
     Dx[i,:] = field[:3]

     field = RBDY3_GetBodyVector('V____',i+1)
     Vx[i,:] = field[:3]
     tmp = frame.T.dot(field[3:6])
     Wz[i,:] = tmp

     field = RBDY3_GetBodyVector('Fext_',i+1)
     Fx[i,:] = field[:3]
     tmp = frame.T.dot(field[3:6])
     Mz[i,:] = tmp

     field = RBDY3_GetBodyVector('Reac_',i+1)
     Rx[i,:] = field[:3]
     tmp = frame.T.dot(field[3:6])
     Rz[i,:] = tmp

     alpha[i,:] = frame[0,:]
     beta[i,:]  = frame[1,:]
     gamma[i,:] = frame[2,:]

   # les champs user
   if len(kw) != 0:
     for name,value in list(kw.items()):
       addUserField(name, value, pointData, 3)

   # threshold sur la visibilite
   tg = thresholdVisible(ug)

   return tg


############################
# mecamailx

def addCellsToUnstructuredGrid(ele, ug, ns, dim):
    #ns stands for NodeShift

    idx=1

    for j in range(ele[0]):

       #print j,idx,ele[idx]
       if dim == 2:
         if ele[idx] == 4:
           xx = vtk.vtkQuad()
         elif ele[idx] == 8:
           xx = vtk.vtkQuadraticQuad()
         elif ele[idx] == 3:
           xx = vtk.vtkTriangle()
         elif ele[idx] == 6:
           xx = vtk.vtkQuadraticTriangle()
         elif ele[idx] == 2:
           xx = vtk.vtkLine()
         elif ele[idx] == 1:
           xx = vtk.vtkVertex()
         else:
           print('vtk_display:: unsupported kind of element')
           sys.exit(0)
       elif dim == 3:
         if ele[idx] == 4:
           xx = vtk.vtkTetra()
         elif ele[idx] == 10:
           xx = vtk.vtkQuadraticTetra()
         elif ele[idx] == 6:
           xx = vtk.vtkWedge()
         elif ele[idx] == 12:
           xx = vtk.vtkQuadraticLinearWedge()
         elif ele[idx] == 15:
           xx = vtk.vtkQuadraticWedge()
         elif ele[idx] == 8:
           xx = vtk.vtkHexahedron()
         elif ele[idx] == 20:
           xx = vtk.vtkQuadraticHexahedron()
         elif ele[idx] == 2:
           xx = vtk.vtkLine()
         elif ele[idx] == 1:
           xx = vtk.vtkVertex()
         else:
           print('vtk_display:: unsupported kind of element')
           sys.exit(0)
       else:
         print('vtk_display:: unsupported dimension')
         sys.exit(0)

       #print ele[idx+1:idx+1+ele[idx]]
       if dim==3 and (ele[idx]==6 or ele[idx]==12 or ele[idx]==15):
         for k in range(ele[idx]):
           xx.GetPointIds().SetId(k, ns + ele[idx+1+wedgeMap[k]]-1)
       else:
         for k in range(ele[idx]):
           xx.GetPointIds().SetId(k, ns + ele[idx+1+k]-1)

       ug.InsertNextCell(xx.GetCellType(),xx.GetPointIds())

       idx += 1 + ele[idx]

def InitMecaFeToVTK(dim):

    ug = vtk.vtkUnstructuredGrid()

    nbm = mecaMAILx_GetNbMecaMAILx()

    points = vtk.vtkPoints()
    ug.SetPoints(points)

    nbNodes = 0
    nbElems = 0
    for i in range(1, nbm+1 ,1):

        coor = mecaMAILx_GetCooref(i)
        ele  = mecaMAILx_GetConnectivity(i)

        if dim==2 :
          for x,y in coor:
            points.InsertNextPoint(x, y, 0.0)
        elif dim==3:
          for x,y,z in coor:
            points.InsertNextPoint(x, y, z)

        addCellsToUnstructuredGrid(ele, ug, nbNodes, dim)

        #cumulating number of nodes
        nbNodes += coor.shape[0]
        nbElems += ele[0]


    pointData = ug.GetPointData()

    Ids = vtk.vtkIntArray()
    Ids.SetNumberOfComponents(1)
    Ids.SetNumberOfTuples(nbNodes)
    Ids.SetName("Ids")
    pointData.AddArray(Ids)

    Nds = vtk.vtkIntArray()
    Nds.SetNumberOfComponents(1)
    Nds.SetNumberOfTuples(nbNodes)
    Nds.SetName("Node Id")
    pointData.AddArray(Nds)

    Els = vtk.vtkIntArray()
    Els.SetNumberOfComponents(1)
    Els.SetNumberOfTuples(nbElems)
    Els.SetName("Element Id")
    ug.GetCellData().AddArray(Els)

    inode = 0
    ielem = 0
    for i in range( 1, nbm+1 ):
        for n in range( mecaMAILx_GetNbNodes(i) ) :
          Ids.SetTuple1(inode, i)
          Nds.SetTuple1(inode, n+1)
          inode += 1
        for n in range( mecaMAILx_GetNbElements(i) ) :
          Els.SetTuple1(ielem, n+1)
          ielem += 1

    Vis = vtk.vtkIntArray()
    Vis.SetNumberOfComponents(1)
    Vis.SetNumberOfTuples(nbNodes)
    Vis.SetName("Visible")
    pointData.AddArray(Vis)

    listOfFields = {"Disp"   : 3,
                    "Velocy" : 3,
                    "E"      : 9,
                    "E vol"  : 1,
                    "S"      : 9,
                    "Svm"    : 1,
                    "Fext"   : 3,
                    "Fint"   : 3,
                    "Reac"   : 3,
                    "Fdyn"   : 3,
                    "Res"    : 3,
                   }

    for name, size in listOfFields.items():
        f = vtk.vtkFloatArray()
        f.SetNumberOfComponents(size)
        f.SetNumberOfTuples(nbNodes)
        f.SetName(name)
        pointData.AddArray(f)

    return ug

def InitMecaGpToVTK(field):

    pdata = vtk.vtkPolyData()
    pdata.Allocate(1)

    # couting total number of gpv
    nbm = mecaMAILx_GetNbMecaMAILx()
    nbg = 0
    for i_bdyty in range(1, nbm+1):
      for i_blmty in range(1, mecaMAILx_GetNbElements(i_bdyty)+1):
        nbg += mecaMAILx_GetNbGp(i_bdyty, i_blmty)

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nbg)

    #verts  = vtk.vtkCellArray()
    #verts.SetNumberOfCells(nbg)

    #nbg = 0
    #for i_bdyty in range(1, nbm+1):
    #  for i_blmty in range(1, mecaMAILx_GetNbElements(i_bdyty)+1):
    #    for i_gp in range(1, mecaMAILx_GetNbGp(i_bdyty, i_blmty)+1):
    #      vx = vtk.vtkVertex()
    #      vx.GetPointIds().SetId(0, nbg)
    #      verts.InsertNextCell(vx)
    #      nbg += 1

    # link container of points/vertices to polydata
    pdata.SetPoints(points)
    #pdata.SetVerts(verts)

    # adding field to cells/points of polydata
    #cellData  = pdata.GetCellData()
    pointData = pdata.GetPointData()

    # ibdyty integer scalar field
    ibdyty = vtk.vtkIntArray()
    ibdyty.SetNumberOfComponents(1)
    ibdyty.SetNumberOfTuples(nbg)
    ibdyty.SetName("ibdyty")

    # iblmty integer scalar field
    iblmty = vtk.vtkIntArray()
    iblmty.SetNumberOfComponents(1)
    iblmty.SetNumberOfTuples(nbg)
    iblmty.SetName("iblmty")

    nbg = 0
    for i_bdyty in range(1, nbm+1):
      for i_blmty in range(1, mecaMAILx_GetNbElements(i_bdyty)+1):
        for i_gp in range(1, mecaMAILx_GetNbGp(i_bdyty, i_blmty)+1):
          ibdyty.SetTuple1(nbg, i_bdyty)
          iblmty.SetTuple1(nbg, i_blmty)
          nbg += 1

    pointData.AddArray(ibdyty)
    pointData.AddArray(iblmty)

    if field == 'stress':
      listOfFields = {"S_ev1" : 3,
                      "S_ev2" : 3,
                      "S_ev3" : 3,
                      "S1"  : 1,
                      "S2"  : 1,
                      "S3"  : 1,
                     }
    elif field == 'strain':
      listOfFields = {"E_ev1" : 3,
                      "E_ev2" : 3,
                      "E_ev3" : 3,
                      "E1"    : 1,
                      "E2"    : 1,
                      "E3"    : 1,
                     }

    for name, size in listOfFields.items():
        f = vtk.vtkFloatArray()
        f.SetNumberOfComponents(size)
        f.SetNumberOfTuples(nbg)
        f.SetName(name)
        pointData.AddArray(f)

    return pdata

def InitMecaRToVTK(dim):

    pdata = vtk.vtkPolyData()
    pdata.Allocate(1)

    # couting total number of mecaMAILx which are rigid/coro
    nbm = mecaMAILx_GetNbMecaMAILx()
    nbg = 0
    for i_bdyty in range(1, nbm+1):
      if mecaMAILx_IsRigid(i_bdyty):
        nbg += 1

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nbg)

    nbg = 0
    if dim == 2:
      for i_bdyty in range(1, nbm+1):
        if mecaMAILx_IsRigid(i_bdyty):
          coor = mecaMAILx_GetBodyRVector('Coor0', i_bdyty)
          points.SetPoint(nbg, coor[0], coor[1], 0.)
          nbg += 1
    elif dim == 3:
      for i_bdyty in range(1, nbm+1):
        if mecaMAILx_IsRigid(i_bdyty):
          coor = mecaMAILx_GetBodyRVector('Coor0', i_bdyty)
          points.SetPoint(nbg, coor[0], coor[1], coor[2])
          nbg += 1

    # link container of points/vertices to polydata
    pdata.SetPoints(points)

    # adding field to cells/points of polydata
    pointData = pdata.GetPointData()

    # ibdyty integer scalar field
    ibdyty = vtk.vtkIntArray()
    ibdyty.SetNumberOfComponents(1)
    ibdyty.SetNumberOfTuples(nbg)
    ibdyty.SetName("ibdyty")

    nbg = 0
    for i_bdyty in range(1, nbm+1):
      if mecaMAILx_IsRigid(i_bdyty):
        ibdyty.SetTuple1(nbg, i_bdyty)
        nbg += 1

    pointData.AddArray(ibdyty)

    listOfFields = {"Disp"   : 3,
                    "Velocy" : 3,
                    "Reac"   : 3,
                    "Fext"   : 3,
                     }
    if dim==2 :
      listOfFields["Spin Z"]   = 1
      listOfFields["Mext Z"]   = 1
      listOfFields["Torque Z"] = 1
      listOfFields["Rot Z"]    = 1
    else:
      listOfFields["Spin"]   = 3
      listOfFields["Mext"]   = 3
      listOfFields["Torque"] = 3
      listOfFields["alpha"]  = 3
      listOfFields["beta"]   = 3
      listOfFields["gamma"]  = 3

    for name, size in listOfFields.items():
        f = vtk.vtkFloatArray()
        f.SetNumberOfComponents(size)
        f.SetNumberOfTuples(nbg)
        f.SetName(name)
        pointData.AddArray(f)

    return pdata

def pushMecafe(Append,dim,ug,kw):

   init_vtk()

   #print 'managing: mecafe'

   nbm = mecaMAILx_GetNbMecaMAILx() 

   pointData = ug.GetPointData()

   Vis = ns.vtk_to_numpy( pointData.GetArray("Visible") )
   Dx  = ns.vtk_to_numpy( pointData.GetArray("Disp"   ) )
   Vx  = ns.vtk_to_numpy( pointData.GetArray("Velocy" ) )
   Exx = ns.vtk_to_numpy( pointData.GetArray("E"      ) )
   Evol= ns.vtk_to_numpy( pointData.GetArray("E vol"  ) )
   Sxx = ns.vtk_to_numpy( pointData.GetArray("S"      ) )
   Svm = ns.vtk_to_numpy( pointData.GetArray("Svm"    ) )
   Fe  = ns.vtk_to_numpy( pointData.GetArray("Fext"   ) )
   Fi  = ns.vtk_to_numpy( pointData.GetArray("Fint"   ) )
   Re  = ns.vtk_to_numpy( pointData.GetArray("Reac"   ) )
   Fd  = ns.vtk_to_numpy( pointData.GetArray("Fdyn"   ) )
   Fr  = ns.vtk_to_numpy( pointData.GetArray("Res"    ) )

   inode = 0

   Vis[:] = 1
   for i in range(1,nbm+1,1):
     
     if not mecaMAILx_IsVisible(i) : 
       for n in range( mecaMAILx_GetNbNodes(i) ):
         Vis[inode] = 0
         inode += 1
       continue

     All  = mecaMAILx_GetAll(i)
     nb_nodes = All.shape[0]
     
     if dim==2:
       Dx[ inode:inode+nb_nodes, :2] = All[:, 0:2 ]
       Dx[ inode:inode+nb_nodes, 2 ] = 0.
       Vx[ inode:inode+nb_nodes, :2] = All[:, 2:4 ]
       Exx[inode:inode+nb_nodes, 0 ] = All[:,  4  ]
       Exx[inode:inode+nb_nodes, 1 ] = All[:,  6  ]
       Exx[inode:inode+nb_nodes, 2 ] = 0.
       Exx[inode:inode+nb_nodes, 3 ] = All[:,  6  ]
       Exx[inode:inode+nb_nodes, 4 ] = All[:,  5  ]
       Exx[inode:inode+nb_nodes,5:8] = 0.
       Exx[inode:inode+nb_nodes, 8 ] = All[:,  7  ]
       Evol[inode:inode+nb_nodes]    = All[:,  8  ]
       Sxx[inode:inode+nb_nodes, 0 ] = All[:,  9  ]
       Sxx[inode:inode+nb_nodes, 1 ] = All[:, 11  ]
       Sxx[inode:inode+nb_nodes, 2 ] = 0.
       Sxx[inode:inode+nb_nodes, 3 ] = All[:, 11  ]
       Sxx[inode:inode+nb_nodes, 4 ] = All[:, 10  ]
       Sxx[inode:inode+nb_nodes,5:8] = 0.
       Sxx[inode:inode+nb_nodes, 8 ] = All[:, 12  ]
       Svm[inode:inode+nb_nodes]     = All[:, 13  ]
       Fe[ inode:inode+nb_nodes, :2] = All[:,14:16]
       Fe[ inode:inode+nb_nodes, 2 ] = 0.
       Fi[ inode:inode+nb_nodes, :2] = All[:,16:18]
       Fi[ inode:inode+nb_nodes, 2 ] = 0.
       Re[ inode:inode+nb_nodes, :2] = All[:,18:20]
       Re[ inode:inode+nb_nodes, 2 ] = 0.
       Fd[ inode:inode+nb_nodes, :2] = All[:,20:22]
       Fd[ inode:inode+nb_nodes, 2 ] = 0.
       Fr[ inode:inode+nb_nodes, :2] = All[:,22:24]
       Fr[ inode:inode+nb_nodes, 2 ] = 0.

     elif dim == 3:

       Dx[ inode:inode+nb_nodes, : ] = All[:, 0:3 ]
       Vx[ inode:inode+nb_nodes, : ] = All[:, 3:6 ]
       Exx[inode:inode+nb_nodes, 0 ] = All[:,  6  ]
       Exx[inode:inode+nb_nodes, 1 ] = All[:,  7  ]
       Exx[inode:inode+nb_nodes, 2 ] = All[:,  9  ]
       Exx[inode:inode+nb_nodes, 3 ] = All[:,  7  ]
       Exx[inode:inode+nb_nodes, 4 ] = All[:,  8  ]
       Exx[inode:inode+nb_nodes, 5 ] = All[:, 10  ]
       Exx[inode:inode+nb_nodes, 6 ] = All[:,  9  ]
       Exx[inode:inode+nb_nodes, 7 ] = All[:, 10  ]
       Exx[inode:inode+nb_nodes, 8 ] = All[:, 11  ]
       Evol[inode:inode+nb_nodes]    = All[:, 12  ]
       Sxx[inode:inode+nb_nodes, 0 ] = All[:, 13  ]
       Sxx[inode:inode+nb_nodes, 1 ] = All[:, 14  ]
       Sxx[inode:inode+nb_nodes, 2 ] = All[:, 16  ]
       Sxx[inode:inode+nb_nodes, 3 ] = All[:, 14  ]
       Sxx[inode:inode+nb_nodes, 4 ] = All[:, 15  ]
       Sxx[inode:inode+nb_nodes, 5 ] = All[:, 17  ]
       Sxx[inode:inode+nb_nodes, 6 ] = All[:, 16  ]
       Sxx[inode:inode+nb_nodes, 7 ] = All[:, 17  ]
       Sxx[inode:inode+nb_nodes, 8 ] = All[:, 18  ]
       Svm[inode:inode+nb_nodes]     = All[:, 19  ]
       Fe[ inode:inode+nb_nodes, : ] = All[:,20:23]
       Fi[ inode:inode+nb_nodes, : ] = All[:,23:26]
       Re[ inode:inode+nb_nodes, : ] = All[:,26:29]
       Fd[ inode:inode+nb_nodes, : ] = All[:,29:32]
       Fr[ inode:inode+nb_nodes, : ] = All[:,32:35]

     inode += nb_nodes

   if len(kw) != 0:
     for name,value in list(kw.items()):
       if value[0] == 'element':
         data = ug.GetCellData()
       elif value[0] == 'node':
         data = ug.GetPointData()
       else:
         print("[pushMecafe] user field should specify element or node... skipping")
         continue

       addUserField(name, value[1], data, dim)


   # threshold sur la visibilite
   tg  = thresholdVisible(ug)
   nbm = tg.GetNumberOfCells()

   Append.AddInputData(tg)

   return nbm

def pushMecaGp(Append,dim,pdata,field):

   init_vtk()

   #print 'managing: mecagp'

   nbm = mecaMAILx_GetNbMecaMAILx()

   points    = pdata.GetPoints()
   pointData = pdata.GetPointData()
   #cellData  = pdata.GetCellData()

   nbg = 0

   # setting points coordinages
   if dim == 2 :

       for i_bdyty in range(1, nbm+1):
         gp_coor = mecaMAILx_GetGpCoor(i_bdyty)
         for i_gp, coor in enumerate(gp_coor):
           points.SetPoint(nbg+i_gp, coor[0], coor[1], 0)
         nbg += gp_coor.shape[0]

   else :

       for i_bdyty in range(1, nbm+1):
         gp_coor = mecaMAILx_GetGpCoor(i_bdyty)
         for i_gp, coor in enumerate(gp_coor):
           points.SetPoint(nbg+i_gp, coor[0], coor[1], coor[2])
         nbg += gp_coor.shape[0]


   # setting stress tensor and principal value
   if field == 'stress':
     i_field = 2
     ev1 = pointData.GetArray("S_ev1" )
     ev2 = pointData.GetArray("S_ev2" )
     ev3 = pointData.GetArray("S_ev3" )
     V1  = pointData.GetArray("S1" )
     V2  = pointData.GetArray("S2" )
     V3  = pointData.GetArray("S3" )
   elif field == 'strain':
     i_field = 1
     ev1 = pointData.GetArray("E_ev1" )
     ev2 = pointData.GetArray("E_ev2" )
     ev3 = pointData.GetArray("E_ev3" )
     V1  = pointData.GetArray("E1" )
     V2  = pointData.GetArray("E2" )
     V3  = pointData.GetArray("E3" )

   nbg = 0
   for i_bdyty in range(1, nbm+1):
     for i_blmty in range(1, mecaMAILx_GetNbElements(i_bdyty)+1):
       for i_gp in range(1, mecaMAILx_GetNbGp(i_bdyty, i_blmty)+1):
         tensor = mecaMAILx_GetGpPrincipalField(i_bdyty, i_blmty, i_gp, i_field)
         V1.SetTuple1(nbg ,  tensor[0,0])
         V2.SetTuple1(nbg ,  tensor[0,1])
         V3.SetTuple1(nbg ,  tensor[0,2])
         ev1.SetTuple3(nbg, *tensor[1,:])
         ev2.SetTuple3(nbg, *tensor[2,:])
         ev3.SetTuple3(nbg, *tensor[3,:])
         nbg += 1

   Append.AddInputData(pdata)

   return nbm

def pushMecaR(Append,dim,pdata,kw):

   init_vtk()

   #print 'managing: mecagp'

   nbm = mecaMAILx_GetNbMecaMAILx()

   points    = pdata.GetPoints()
   pointData = pdata.GetPointData()

   nbg = 0

   # update fields value
   Dx  = pointData.GetArray("Disp"   )
   Vx  = pointData.GetArray("Velocy" )
   Fe  = pointData.GetArray("Fext"   )
   Re  = pointData.GetArray("Reac"   )
   if dim==2 :
     ro  = pointData.GetArray("Rot Z"   )
     sp  = pointData.GetArray("Spin Z"  )
     me  = pointData.GetArray("Mext Z"  )
     to  = pointData.GetArray("Torque Z")
   else:
     sp  = pointData.GetArray("Spin"   )
     me  = pointData.GetArray("Mext"   )
     to  = pointData.GetArray("Torque" )
     al  = pointData.GetArray("alpha"  )
     be  = pointData.GetArray("beta"   )
     ga  = pointData.GetArray("gamma"  )

   nbg = 0
   if dim == 2:
     for i_bdyty in range(1, nbm+1):
       if mecaMAILx_IsRigid(i_bdyty):
         disp = mecaMAILx_GetBodyRVector('X____', i_bdyty)
         velo = mecaMAILx_GetBodyRVector('V____', i_bdyty)
         fext = mecaMAILx_GetBodyRVector('Fext_', i_bdyty)
         reac = mecaMAILx_GetBodyRVector('Reac_', i_bdyty)
         Dx.SetTuple3(nbg , disp[0], disp[1], 0.)
         Vx.SetTuple3(nbg , velo[0], velo[1], 0.)
         Fe.SetTuple3(nbg , fext[0], fext[1], 0.)
         Re.SetTuple3(nbg , reac[0], reac[1], 0.)
         ro.SetTuple1(nbg , disp[2])
         sp.SetTuple1(nbg , velo[2])
         me.SetTuple1(nbg , fext[2])
         to.SetTuple1(nbg , reac[2])
         nbg += 1
   elif dim == 3:
     for i_bdyty in range(1, nbm+1):
       if mecaMAILx_IsRigid(i_bdyty):
         disp = mecaMAILx_GetBodyRVector('X____', i_bdyty)
         velo = mecaMAILx_GetBodyRVector('V____', i_bdyty)
         reac = mecaMAILx_GetBodyRVector('Reac_', i_bdyty)
         fext = mecaMAILx_GetBodyRVector('Fext_', i_bdyty)
         fram = mecaMAILx_GetRigidFrame('RF___', i_bdyty)
         Dx.SetTuple3(nbg , disp[0], disp[1], disp[2])
         Vx.SetTuple3(nbg , velo[0], velo[1], velo[2])
         Re.SetTuple3(nbg , reac[0], reac[1], reac[2])
         Fe.SetTuple3(nbg , fext[0], fext[1], fext[2])
         sp.SetTuple3(nbg , velo[3], velo[4], velo[5])
         me.SetTuple3(nbg , fext[3], fext[4], fext[5])
         to.SetTuple3(nbg , reac[3], reac[4], reac[5])
         al.SetTuple3(nbg , fram[0,0], fram[0,1], fram[0,2])
         be.SetTuple3(nbg , fram[1,0], fram[1,1], fram[1,2])
         ga.SetTuple3(nbg , fram[2,0], fram[2,1], fram[2,2])
         nbg += 1

   if len(kw) != 0:
     for name,value in list(kw.items()):
       addUserField(name, value[1], pointData, dim)

   Append.AddInputData(pdata)

   return nbm


def addUserField(name, val, data, dim):

    init_vtk()

    uField = data.GetArray(name)

    #creating vtk array
    if uField is None :

      if isinstance( val, list ):

        datatype = val[0].dtype.type
        if val[0].ndim > 2 :
          print("[addUserField] cannot handle tensor field "+name+"... skipping")
          return
        elif val[0].ndim > 1 :
          nb_comp = val[0].shape[1]
          nb_comp = 3 if nb_comp == 2 and dim ==2 else nb_comp
        else :
          nb_comp = 1

        nb_tuples = 0
        for v in val:
          nb_tuples += v.shape[0]

      else:

        datatype = val.dtype.type
        if val.ndim > 2 :
          print("[addUserField] cannot handle tensor field "+name+"... skipping")
          return
        elif val.ndim > 1 :
          nb_comp = val.shape[1]
          #in case of 3d vector needed in 2d space
          nb_comp = 3 if nb_comp == 2 and dim ==2 else nb_comp
        else :
          nb_comp = 1

        nb_tuples = val.shape[0]

      # it is important to test float...
      # because np.int is not int whereas np.float is a float !
      if issubclass( datatype, float ):
        uField = vtk.vtkFloatArray()
      else :
        uField = vtk.vtkIntArray()

      uField.SetName(name)
      uField.SetNumberOfComponents( nb_comp )
      uField.SetNumberOfTuples( nb_tuples )
      data.AddArray(uField)

    vField = ns.vtk_to_numpy( uField )

    #filling vtk array
    if isinstance( val, list ):
      offset = 0
      for v in val:
        vsize = v.shape[0]
        if v.ndim == 1 :
          vField[offset:offset+vsize] = v[:]
        else:
          #in case of 3d vector needed in 2d space
          if dim==2 and v.shape[1]==2 :
            vField[offset:offset+vsize,:dim] = v[:,:dim]
            vField[offset:offset+vsize, dim] = 0
          else:
            vField[offset:offset+vsize,:] = v[:,:]
        offset += vsize

    else :
      if val.ndim == 1 :
        vField[:] = val[:]
      else:
        if dim==2 and val.shape[1]==2 :
          vField[:,:dim] = val[:,:dim]
          vField[:, dim] = 0
        else:
          vField[:,:] = val[:,:]


############################
# thermailx

def InitTherFeToVTK(dim):

    ug = vtk.vtkUnstructuredGrid()

    nbm = therMAILx_GetNbTherMAILx()

    points = vtk.vtkPoints()
    ug.SetPoints(points)

    nbNodes = 0
    nbElems = 0
    for i in range(1, nbm+1 ,1):

        coor = therMAILx_GetCoor(i)
        ele  = therMAILx_GetConnectivity(i)

        if dim==2 :
          for x,y in coor:
            points.InsertNextPoint(x, y, 0.0)
        elif dim==3:
          for x,y,z in coor:
            points.InsertNextPoint(x, y, z)

        addCellsToUnstructuredGrid(ele, ug, nbNodes, dim)

        #cumulating number of nodes
        nbNodes += coor.shape[0]
        nbElems += ele[0]

    pointData = ug.GetPointData()

    Ids = vtk.vtkIntArray()
    Ids.SetNumberOfComponents(1)
    Ids.SetNumberOfTuples(nbNodes)
    Ids.SetName("Ids")
    pointData.AddArray(Ids)

    Nds = vtk.vtkIntArray()
    Nds.SetNumberOfComponents(1)
    Nds.SetNumberOfTuples(nbNodes)
    Nds.SetName("Node Id")
    pointData.AddArray(Nds)

    Els = vtk.vtkIntArray()
    Els.SetNumberOfComponents(1)
    Els.SetNumberOfTuples(nbElems)
    Els.SetName("Element Id")
    ug.GetCellData().AddArray(Els)

    inode = 0
    ielem = 0
    for i in range( 1, nbm+1 ):
        for n in range( therMAILx_GetNbNodes(i) ) :
          Ids.SetTuple1(inode, i)
          Nds.SetTuple1(inode, n+1)
          inode += 1
        for n in range( therMAILx_GetNbElements(i) ) :
          Els.SetTuple1(ielem, n+1)
          ielem += 1

    #Vis = vtk.vtkIntArray()
    #Vis.SetNumberOfComponents(1)
    #Vis.SetNumberOfTuples(nbNodes)
    #Vis.SetName("Visible")
    #pointData.AddArray(Vis)

    listOfFields = {"Temperature" : 1,
                    "Grad"        : 3,
                    "Flux"        : 3,
                   }

    for name, size in listOfFields.items():
        f = vtk.vtkFloatArray()
        f.SetNumberOfComponents(size)
        f.SetNumberOfTuples(nbNodes)
        f.SetName(name)
        pointData.AddArray(f)

    return ug

def InitTherGpToVTK(field):

    pdata = vtk.vtkPolyData()
    pdata.Allocate(1)

    # couting total number of gpv
    nbm = therMAILx_GetNbTherMAILx()
    nbg = 0
    for i_bdyty in range(1, nbm+1):
      for i_blmty in range(1, therMAILx_GetNbElements(i_bdyty)+1):
        nbg += therMAILx_GetNbGp(i_bdyty, i_blmty)

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nbg)

    # link container of points/vertices to polydata
    pdata.SetPoints(points)

    # adding field to points of polydata
    pointData = pdata.GetPointData()

    # ibdyty integer scalar field
    ibdyty = vtk.vtkIntArray()
    ibdyty.SetNumberOfComponents(1)
    ibdyty.SetNumberOfTuples(nbg)
    ibdyty.SetName("ibdyty")

    # iblmty integer scalar field
    iblmty = vtk.vtkIntArray()
    iblmty.SetNumberOfComponents(1)
    iblmty.SetNumberOfTuples(nbg)
    iblmty.SetName("iblmty")

    nbg = 0
    for i_bdyty in range(1, nbm+1):
      for i_blmty in range(1, therMAILx_GetNbElements(i_bdyty)+1):
        for i_gp in range(1, therMAILx_GetNbGp(i_bdyty, i_blmty)+1):
          ibdyty.SetTuple1(nbg, i_bdyty)
          iblmty.SetTuple1(nbg, i_blmty)
          nbg += 1

    pointData.AddArray(ibdyty)
    pointData.AddArray(iblmty)

    listOfFields = {f : 3 for f in field }

    for name, size in listOfFields.items():
        f = vtk.vtkFloatArray()
        f.SetNumberOfComponents(size)
        f.SetNumberOfTuples(nbg)
        f.SetName(name)
        pointData.AddArray(f)

    return pdata

def pushTherfe(Append,dim,ug,kw):

   init_vtk()

   #print 'managing: therfe'

   nbm = therMAILx_GetNbTherMAILx()

   pointData = ug.GetPointData()

   #Vis  = pointData.GetArray("Visible"    )
   T    = pointData.GetArray("Temperature")
   Flux = pointData.GetArray("Flux"       )
   Grad = pointData.GetArray("Grad"       )

   inode = 0

   for i in range(1,nbm+1,1):
     
     #if not therMAILx_IsVisible(i) :
     #  for n in range( therMAILx_GetNbNodes(i) ):
     #    Vis.SetTuple1(inode, 0)
     #    inode += 1
     #  continue

     All  = therMAILx_GetAll(i)

     for tt,gx,gy,gz,fx,fy,fz in All:
        #Vis.SetTuple1(inode, 1)
        T.SetTuple1(inode, tt)
        Flux.SetTuple3(inode, fx, fy, fz)
        Grad.SetTuple3(inode, gx, gy, gz)

        inode += 1

   if len(kw) != 0:
     for name,value in list(kw.items()):
       if value[0] == 'element':
         data = ug.GetCellData()
       elif value[0] == 'node':
         data = ug.GetPointData()
       else:
         print("[pushTherfe] user field should specify element or node... skipping")
         continue

       addUserField(name, value[1], data, dim)

   ##  no visibility management with therFE
   ### threshold sur la visibilite
   ##threshold = vtk.vtkThreshold()
   ##threshold.SetInputData(ug)
   ##threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Visible");
   ##threshold.setUpperThreshold(1)
   ##threshold.Update()
   ##tg = threshold.GetOutput()
   ##tg.GetPointData().RemoveArray("Visible")

   Append.AddInputData(ug)

   return nbm

def pushTherGp(Append,dim,pdata,field):

   init_vtk()

   #print 'managing: mecagp'

   nbm = therMAILx_GetNbTherMAILx()

   points    = pdata.GetPoints()
   pointData = pdata.GetPointData()

   nbg = 0

   # field name to index in lmgc90:
   f2i = { 'gradT' : 1, 'fluxT' : 2 }
   # setting points coordinages
   if dim == 2 :

       for i_bdyty in range(1, nbm+1):
         gp_coor = therMAILx_GetGpCoor(i_bdyty)
         for i_gp, coor in enumerate(gp_coor):
           points.SetPoint(nbg+i_gp, coor[0], coor[1], 0)
         nbg += gp_coor.shape[0]

   else :

       for i_bdyty in range(1, nbm+1):
         gp_coor = therMAILx_GetGpCoor(i_bdyty)
         for i_gp, coor in enumerate(gp_coor):
           points.SetPoint(nbg+i_gp, coor[0], coor[1], coor[2])
         nbg += gp_coor.shape[0]


   # setting stress tensor and principal value
   field = { f:pointData.GetArray(f) for f in field }

   nbg = 0
   for i_bdyty in range(1, nbm+1):
     for i_blmty in range(1, therMAILx_GetNbElements(i_bdyty)+1):
       for i_gp in range(1, therMAILx_GetNbGp(i_bdyty, i_blmty)+1):
         for n, f in field.items():
           value = therMAILx_GetGpField(i_bdyty, i_blmty, i_gp, f2i[n])
           f.SetTuple3(nbg, *value)
         nbg += 1

   Append.AddInputData(pdata)

   return nbm


############################
# poromailx

def InitPoroFeToVTK(dim):

    ug = vtk.vtkUnstructuredGrid()

    nbm = poroMAILx_GetNbPoroMAILx()

    points = vtk.vtkPoints()
    ug.SetPoints(points)

    nbNodes = 0
    nbElems = 0
    for i in range(1, nbm+1 ,1):

        coor = poroMAILx_GetCoor(i)
        ele  = poroMAILx_GetConnectivity(i)

        if dim==2 :
          for x,y in coor:
            points.InsertNextPoint(x, y, 0.0)
        elif dim==3:
          for x,y,z in coor:
            points.InsertNextPoint(x, y, z)

        addCellsToUnstructuredGrid(ele, ug, nbNodes, dim)

        #cumulating number of nodes
        nbNodes += coor.shape[0]
        nbElems += ele[0]


    pointData = ug.GetPointData()

    Ids = vtk.vtkIntArray()
    Ids.SetNumberOfComponents(1)
    Ids.SetNumberOfTuples(nbNodes)
    Ids.SetName("Ids")
    pointData.AddArray(Ids)

    Nds = vtk.vtkIntArray()
    Nds.SetNumberOfComponents(1)
    Nds.SetNumberOfTuples(nbNodes)
    Nds.SetName("Node Id")
    pointData.AddArray(Nds)

    Els = vtk.vtkIntArray()
    Els.SetNumberOfComponents(1)
    Els.SetNumberOfTuples(nbElems)
    Els.SetName("Element Id")
    ug.GetCellData().AddArray(Els)

    inode = 0
    ielem = 0
    for i in range( 1, nbm+1 ):
        for n in range( poroMAILx_GetNbNodes(i) ) :
          Ids.SetTuple1(inode, i)
          Nds.SetTuple1(inode, n+1)
          inode += 1
        for n in range( poroMAILx_GetNbElements(i) ) :
          Els.SetTuple1(ielem, n+1)
          ielem += 1

    Vis = vtk.vtkIntArray()
    Vis.SetNumberOfComponents(1)
    Vis.SetNumberOfTuples(nbNodes)
    Vis.SetName("Visible")
    pointData.AddArray(Vis)

    listOfFields = {"Almansi Strain" : 9,
                    "Cauchy Stress"  : 9,
                    "Disp"           : 3,
                    "Velocity"       : 3,
                    "Grad P"         : 3,
                    "Darcy Flux"     : 3,
                    "Jacobien"       : 1,
                    "Svm"            : 1,
                    "Pressure"       : 1,
                    "Fext"           : 3,
                    "Fint"           : 3,
                   }

    for name, size in listOfFields.items():
        f = vtk.vtkFloatArray()
        f.SetNumberOfComponents(size)
        f.SetNumberOfTuples(nbNodes)
        f.SetName(name)
        pointData.AddArray(f)

    return ug

def pushPorofe(Append,dim,ug,kw):

   init_vtk()

   #print 'managing: porofe'

   nbm = poroMAILx_GetNbPoroMAILx()

   pointData = ug.GetPointData()

   Vis      = pointData.GetArray("Visible"       )
   Strain   = pointData.GetArray("Almansi Strain")
   Stress   = pointData.GetArray("Cauchy Stress" )
   Disp     = pointData.GetArray("Disp"          )
   Velocity = pointData.GetArray("Velocity"      )
   Grad     = pointData.GetArray("Grad P"        )
   Flux     = pointData.GetArray("Darcy Flux"    )
   Evol     = pointData.GetArray("Jacobien"      )
   Svm      = pointData.GetArray("Svm"           )
   Pressure = pointData.GetArray("Pressure"      )
   Fe       = pointData.GetArray("Fext"          )
   Fi       = pointData.GetArray("Fint"          )

   inode = 0

   for i in range(1,nbm+1,1):

     All  = poroMAILx_GetAll(i)

     if dim==2:
 
       for dx,dy,vx,vy,p,exx,eyy,exy,ezz,J,sxx,syy,sxy,szz,svm,gradpx,gradpy,gradpz,fluxx,fluxy,fluxz,fex,fey,fix,fiy in All:

          Strain.SetTuple9(inode, exx, exy, 0.0, exy, eyy, 0.0, 0.0, 0.0, ezz)
          Stress.SetTuple9(inode, sxx, sxy, 0.0, sxy, syy, 0.0, 0.0, 0.0, szz)
          Disp.SetTuple3(inode, dx, dy, 0.0)
          Velocity.SetTuple3(inode, vx, vy, 0.0)
          Grad.SetTuple3(inode, gradpx, gradpy, 0.0)
          Flux.SetTuple3(inode, fluxx, fluxy, 0.0)
          Evol.SetTuple1(inode, J)
          Svm.SetTuple1(inode, svm)
          Pressure.SetTuple1(inode, p)
          Fe.SetTuple3(inode,fex,fey,0.)
          Fi.SetTuple3(inode,fix,fiy,0.)

          inode += 1

     elif dim == 3:

       for dx,dy,dz,vx,vy,vz,p,exx,exy,eyy,exz,eyz,ezz,J,sxx,sxy,syy,sxz,syz,szz,svm,gradpx,gradpy,gradpz,fluxx,fluxy,fluxz,fex,fey,fez,fix,fiy,fiz in All:
          Strain.SetTuple9(inode, exx, exy, exz, exy, eyy, eyz, exz, eyz, ezz)
          Stress.SetTuple9(inode, sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz)
          Disp.SetTuple3(inode, dx, dy, dz)
          Velocity.SetTuple3(inode, vx, vy, vz)
          Grad.SetTuple3(inode, gradpx, gradpy, gradpz)
          Flux.SetTuple3(inode, fluxx, fluxy, fluxz)
          Evol.SetTuple1(inode, J)
          Svm.SetTuple1(inode, svm)
          Pressure.SetTuple1(inode, p)
          Fe.SetTuple3(inode,fex,fey,fez)
          Fi.SetTuple3(inode,fix,fiy,fiz)

          inode += 1

   if len(kw) != 0:
     for name,value in list(kw.items()):
       if value[0] == 'element':
         data = ug.GetCellData()
       elif value[0] == 'node':
         data = ug.GetPointData()
       else:
         print("[pushTherfe] user field should specify element or node... skipping")
         continue

       addUserField(name, value[1], data, dim)
        
   Append.AddInputData(ug)

   return nbm

#######################################################
# info solver

def pushThis(Append,dt,lr):

   init_vtk()

   All = nlgs_GetAllThis()
   if All is None or All.size==0:
       return 0

   nb_inter  = All.shape[0]

   mean_fn= np.mean(All[:,7])
   max_fn = np.max( All[:,7])
   min_fn = np.min( All[:,7]) 

   # pour eviter de diviser par la suite apres
   seuil_fn = max( abs(max_fn), abs(min_fn) )

   # pour eviter de diviser par 0. apres
   if seuil_fn == 0.: seuil_fn = 1.

   grid = vtk.vtkPolyData()
   grid.Allocate(nb_inter)

   points = vtk.vtkPoints()
   points.SetNumberOfPoints(2*nb_inter)

   Ids = vtk.vtkIntArray()
   Ids.SetName("Ids")

   Rn = vtk.vtkFloatArray()
   Rn.SetName("Rn")

   Rt = vtk.vtkFloatArray()
   Rt.SetName("Rt")

   Vn = vtk.vtkFloatArray()
   Vn.SetName("Vn")

   Vt = vtk.vtkFloatArray()
   Vt.SetName("Vt")

   Gap= vtk.vtkFloatArray()
   Vt.SetName("Gap")

   Strong = vtk.vtkIntArray()
   Strong.SetName("Strong")

   for f in [Ids, Rt, Rn, Vt, Vn, Gap, Strong]:
       f.SetNumberOfComponents(1)
       f.SetNumberOfTuples(nb_inter)

   for i_cdan in range(nb_inter):
        coorx,coory,tx,ty,nx,ny,ft,fn,vt,vn,gap = All[i_cdan]

        if fn > mean_fn:
          s = 1
        else:
          s = 0

        vertex = buildSegment(coorx,coory,0.,nx,ny,0.,lr)

        segment = vtk.vtkLine()
        for k in range(2):
            x, y, z = vertex[k,:]
            points.SetPoint(2*i_cdan+k, x,y,z)
            segment.GetPointIds().SetId(k, 2*i_cdan+k)

        grid.InsertNextCell(segment.GetCellType(), segment.GetPointIds())

        Ids.SetTuple1(i_cdan,i_cdan+1)
        Rt.SetTuple1(i_cdan,ft/dt)
        Rn.SetTuple1(i_cdan,fn/dt)
        Vt.SetTuple1(i_cdan,vt)
        Vn.SetTuple1(i_cdan,vn)
        Gap.SetTuple1(i_cdan,gap)
        Strong.SetTuple1(i_cdan,s)


   grid.SetPoints(points)
   cellData = grid.GetCellData()

   for f in [Ids, Rt, Rn, Vt, Vn, Strong]:
       cellData.AddArray(f)

   # CellData to PointData
   c2p = vtk.vtkCellDataToPointData()
   c2p.SetInputData(grid)
   c2p.PassCellDataOff()
   c2p.Update()

   Append.AddInputData(c2p.GetPolyDataOutput())

   return nb_inter


def writeThisToVTK(fichier,fid,step,lr,dt=1.):

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writeThisToVTK] Since vtk module is not available this function does nothing' )
      return

   vtpFile = vtk.vtkXMLPolyDataWriter()
   # rm: what is the real use of this ?
   #vtpFile.SetIdTypeToInt32()
   #vtpFile.SetIdTypeToInt64()
   vtpFile.SetFileName(fichier)

   AppendAll=vtk.vtkAppendPolyData()

   nb = pushThis(AppendAll,dt,lr)

   if nb :

     vtpFile.SetInputConnection(AppendAll.GetOutputPort())
     vtpFile.Write()

     impr='<DataSet timestep="%s" group="" part="0" file="%s"/>\n' % (step,'./'+os.path.basename(fichier))
     impr+= '</Collection>\n</VTKFile>'

     removeLines(fid)
     with open(fid,'a') as f:
       f.write(impr)

###################
# multi FE

def InitMultiFeToVTK(dim):

    ug = vtk.vtkUnstructuredGrid()

    nbm = multiMAILx_GetNb()

    points = vtk.vtkPoints()
    ug.SetPoints(points)

    nbNodes = 0
    nbElems = 0
    for i in range(1, nbm+1 ,1):

        coor = multiMAILx_GetCoor(i)
        ele  = multiMAILx_GetConnectivity(i)

        if dim==2 :
          for x,y in coor:
            points.InsertNextPoint(x, y, 0.0)
        elif dim==3:
          for x,y,z in coor:
            points.InsertNextPoint(x, y, z)

        addCellsToUnstructuredGrid(ele, ug, nbNodes, dim)

        #cumulating number of nodes
        nbNodes += coor.shape[0]
        nbElems += ele[0]

    pointData = ug.GetPointData()

    Ids = vtk.vtkIntArray()
    Ids.SetNumberOfComponents(1)
    Ids.SetNumberOfTuples(nbNodes)
    Ids.SetName("Ids")
    pointData.AddArray(Ids)

    Nds = vtk.vtkIntArray()
    Nds.SetNumberOfComponents(1)
    Nds.SetNumberOfTuples(nbNodes)
    Nds.SetName("Node Id")
    pointData.AddArray(Nds)

    Els = vtk.vtkIntArray()
    Els.SetNumberOfComponents(1)
    Els.SetNumberOfTuples(nbElems)
    Els.SetName("Element Id")
    ug.GetCellData().AddArray(Els)

    inode = 0
    ielem = 0
    for i in range( 1, nbm+1 ):
        for n in range( mecaMAILx_GetNbNodes(i) ) :
          Ids.SetTuple1(inode, i)
          Nds.SetTuple1(inode, n+1)
          inode += 1
        for n in range( multiMAILx_GetNbElements(i) ) :
          Els.SetTuple1(ielem, n+1)
          ielem += 1

    Vis = vtk.vtkIntArray()
    Vis.SetNumberOfComponents(1)
    Vis.SetNumberOfTuples(nbNodes)
    Vis.SetName("Visible")
    pointData.AddArray(Vis)

    listOfFields = {"Disp"         : 3,
                    "Velocy"       : 3,
                    "Pc"           : 1,
                    "Pn"           : 1,
                    "Force ext"    : 3,
                    "Force int"    : 3,
                    "Force dmp"    : 3,
                    "Force dyn"    : 3,
                    "Reac"         : 3,
                    "Res"          : 3,
                    "Flux Pc ext"  : 1,
                    "Flux Pc int"  : 1,
                    "Flux Pc dmp"  : 1,
                    "Flux Pc dyn"  : 1,
                    "Reac Pc"      : 1,
                    "Res Pc"       : 1,
                    "Flux Pn ext"  : 1,
                    "Flux Pn int"  : 1,
                    "Flux Pn dmp"  : 1,
                    "Flux Pn dyn"  : 1,
                    "Reac Pn"      : 1,
                    "Res Pn"       : 1,
                    "Strain"       : 9,
                    "Stress"       : 9,
                    "Grad Pc"      : 3,
                    "Pc Darcy Flux": 3,
                    "Grad Pn"      : 3,
                    "Pn Darcy Flux": 3,
                   }

    for name, size in listOfFields.items():
        f = vtk.vtkFloatArray()
        f.SetNumberOfComponents(size)

def pushMultife(Append,dim,ug,kw):

   init_vtk()

   nbm = multiMAILx_GetNb() 

   pointData = ug.GetPointData()

   Vis  = pointData.GetArray("Visible")
   Dx   = pointData.GetArray("Disp")
   Vx   = pointData.GetArray("Velocy")
   Pc   = pointData.GetArray("Pc")
   Pn   = pointData.GetArray("Pn")
   Fe   = pointData.GetArray("Force ext")
   Fi   = pointData.GetArray("Force int")
   Fp   = pointData.GetArray("Force dmp")
   Fd   = pointData.GetArray("Force dyn")
   Re   = pointData.GetArray("Reac")
   Fr   = pointData.GetArray("Res")
   FePc = pointData.GetArray("Flux Pc ext")
   FiPc = pointData.GetArray("Flux Pc int")
   FpPc = pointData.GetArray("Flux Pc dmp")
   FdPc = pointData.GetArray("Flux Pc dyn")
   RePc = pointData.GetArray("Reac Pc")
   FrPc = pointData.GetArray("Res Pc")
   FePn = pointData.GetArray("Flux Pn ext")
   FiPn = pointData.GetArray("Flux Pn int")
   FpPn = pointData.GetArray("Flux Pn dmp")
   FdPn = pointData.GetArray("Flux Pn dyn")
   RePn = pointData.GetArray("Reac Pn")
   FrPn = pointData.GetArray("Res Pn")
   Exx  = pointData.GetArray("Strain")
   Sxx  = pointData.GetArray("Stress")
   GPc  = pointData.GetArray("Grad Pc")
   FPc  = pointData.GetArray("Pc Darcy Flux")
   GPn  = pointData.GetArray("Grad Pn")
   FPn  = pointData.GetArray("Pn Darcy Flux")

   inode = 0

   for i in range(1,nbm+1,1):
     
     if not multiMAILx_IsVisible(i) : 
       for n in range( multiMAILx_GetNbNodes(i) ):
         Vis.SetTuple1(inode, 0)
         inode += 1
       continue

     All  = multiMAILx_GetAll(i)
     
     if dim==2:

       for val in All:
          Vis.SetTuple1(inode, 1)
          Dx.SetTuple3(inode, val[0], val[1], 0.)
          Vx.SetTuple3(inode, val[2], val[3], 0.)
          Pc.SetTuple1(inode, val[4])
          Pn.SetTuple1(inode, val[5])
          Fe.SetTuple3(inode, val[6] , val[7] , 0.)
          Fi.SetTuple3(inode, val[8] , val[9] , 0.)
          Fp.SetTuple3(inode, val[10], val[11], 0.)
          Fd.SetTuple3(inode, val[12], val[13], 0.)
          Re.SetTuple3(inode, val[14], val[15], 0.)
          Fr.SetTuple3(inode, val[16], val[17], 0.)
          FePc.SetTuple1(inode, val[18])
          FiPc.SetTuple1(inode, val[19])
          FpPc.SetTuple1(inode, val[20])
          FdPc.SetTuple1(inode, val[21])
          RePc.SetTuple1(inode, val[22])
          FrPc.SetTuple1(inode, val[23])
          FePn.SetTuple1(inode, val[24])
          FiPn.SetTuple1(inode, val[25])
          FpPn.SetTuple1(inode, val[26])
          FdPn.SetTuple1(inode, val[27])
          RePn.SetTuple1(inode, val[28])
          FrPn.SetTuple1(inode, val[29])
          Exx.SetTuple9(inode, val[30], val[32], 0., val[32], val[31], 0., 0., 0., val[33])
          Sxx.SetTuple9(inode, val[34], val[36], 0., val[36], val[35], 0., 0., 0., val[37])
          GPc.SetTuple3(inode, val[38], val[39], 0.)
          FPc.SetTuple3(inode, val[40], val[41], 0.)
          GPn.SetTuple3(inode, val[42], val[43], 0.)
          FPn.SetTuple3(inode, val[44], val[45], 0.)

          inode += 1

     elif dim == 3:

       for val in All:
          Vis.SetTuple1(inode, 1)
          Dx.SetTuple3(inode, val[0], val[1], val[2])
          Vx.SetTuple3(inode, val[3], val[4], val[5])
          Pc.SetTuple1(inode, val[6])
          Pn.SetTuple1(inode, val[7])
          Fe.SetTuple3(inode, val[8] , val[9] , val[10])
          Fi.SetTuple3(inode, val[11], val[12], val[13])
          Fp.SetTuple3(inode, val[14], val[15], val[16])
          Fd.SetTuple3(inode, val[17], val[18], val[19])
          Re.SetTuple3(inode, val[20], val[21], val[22])
          Fr.SetTuple3(inode, val[23], val[24], val[25])
          FePc.SetTuple1(inode, val[26])
          FiPc.SetTuple1(inode, val[27])
          FpPc.SetTuple1(inode, val[28])
          FdPc.SetTuple1(inode, val[29])
          RePc.SetTuple1(inode, val[30])
          FrPc.SetTuple1(inode, val[31])
          FePn.SetTuple1(inode, val[32])
          FiPn.SetTuple1(inode, val[33])
          FpPn.SetTuple1(inode, val[34])
          FdPn.SetTuple1(inode, val[35])
          RePn.SetTuple1(inode, val[36])
          FrPn.SetTuple1(inode, val[37])
          Exx.SetTuple9(inode, val[30], val[31], val[32], val[31], val[33], val[34], val[32], val[34], val[35])
          Sxx.SetTuple9(inode, val[36], val[37], val[38], val[37], val[39], val[40], val[38], val[40], val[41])
          GPc.SetTuple3(inode, val[42], val[43], val[44])
          FPc.SetTuple3(inode, val[45], val[46], val[47])
          GPn.SetTuple3(inode, val[48], val[49], val[50])
          FPn.SetTuple3(inode, val[51], val[52], val[52])

          inode += 1

   if len(kw) != 0:
     for name,value in list(kw.items()):
       if value[0] == 'element':
         data = ug.GetCellData()
       elif value[0] == 'node':
         data = ug.GetPointData()
       else:
         print("[pushMultife] user field should specify element or node... skipping")
         continue

       addUserField(name, value[1], data, dim)

        
   # threshold sur la visibilite
   #threshold = vtk.vtkThreshold()
   #threshold.SetInputData(ug)
   #threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Visible");
   #threshold.setUpperThreshold(1)
   #threshold.Update()
   #tg = threshold.GetOutput()
   #tg.GetPointData().RemoveArray("Visible")

   Append.AddInputData(ug)

   return nbm


def writeToVTK(fichier,fid,dim,truc,**kw):
   #truc is either a unstructured grid or a tacts_dict

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writeToVTK] Since vtk module is not available this function does nothing' )
      return

   # filename does not hold extension
   # for backward compatibility with vtk_display_old.py
   fichier += '.vtu'

   vtuFile = vtk.vtkXMLUnstructuredGridWriter()
   # rm: what is the real use of this ?
   #vtuFile.SetIdTypeToInt32()
   #vtuFile.SetIdTypeToInt64()
   vtuFile.SetFileName(fichier)
   vtuFile.SetDataMode(vtk.vtkXMLWriter.Binary)

   AppendAll = vtk.vtkAppendFilter()

   # there is probably better
   if 'mecafe' in fichier:
     nb = pushMecafe(AppendAll,dim,truc,kw)
   elif 'therfe' in fichier:
     nb = pushTherfe(AppendAll,dim,truc,kw)
   elif 'porofe' in fichier:
     nb = pushPorofe(AppendAll,dim,truc,kw)
   elif 'multife' in fichier:
     nb = pushMultife(AppendAll,dim,truc,kw)
   elif 'tacts' in fichier:
     if dim == 2:
       nb = pushTactors2D(truc,AppendAll,kw)
     else:
       nb = pushTactors3D(truc,AppendAll,kw)
   elif 'rigids' in fichier:
     if dim == 2:
       AppendAll = pushRigid2D(truc,kw)
     else :
       AppendAll = pushRigid3D(truc,kw)
     nb = AppendAll.GetNumberOfCells()

   if nb :

     if isinstance(AppendAll,vtk.vtkAlgorithm) :
         vtuFile.SetInputConnection(AppendAll.GetOutputPort())
     else:
         vtuFile.SetInputData(AppendAll)

     vtuFile.Write()

     time = TimeEvolution_GetTime()
     impr = '<DataSet timestep="%s" group="" part="0" file="%s"/>\n' % (time,'./'+os.path.basename(fichier))
     impr+= '</Collection>\n</VTKFile>'

     removeLines(fid)
     with open(fid,'a') as f:
       f.write(impr)


def writeGpToVTK(fichier,fid,dim,truc,field):
   #truc is either a unstructured grid or a tacts_dict

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writeToVTK] Since vtk module is not available this function does nothing' )
      return

   fichier += '.vtp'
   vtpFile  = vtk.vtkXMLPolyDataWriter()
   AppendAll = vtk.vtkAppendPolyData()

   # rm: what is the real use of this ?
   #vtpFile.SetIdTypeToInt32()
   #vtpFile.SetIdTypeToInt64()
   vtpFile.SetFileName(fichier)
   vtpFile.SetDataMode(vtk.vtkXMLWriter.Binary)

   if 'mecagp' in fichier:
     nb = pushMecaGp(AppendAll,dim,truc,field)
   elif 'thergp' in fichier:
     nb = pushTherGp(AppendAll,dim,truc,field)

   if nb :

     vtpFile.SetInputConnection(AppendAll.GetOutputPort())
     vtpFile.Write()

     time = TimeEvolution_GetTime()
     impr = '<DataSet timestep="%s" group="" part="0" file="%s"/>\n' % (time,'./'+os.path.basename(fichier))
     impr+= '</Collection>\n</VTKFile>'

     removeLines(fid)
     with open(fid,'a') as f:
       f.write(impr)

def writePdToVTK(fichier,fid,dim,truc,**kw):
   #truc should be a polydata

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writePdToVTK] Since vtk module is not available this function does nothing' )
      return

   fichier += '.vtp'
   vtpFile  = vtk.vtkXMLPolyDataWriter()
   AppendAll = vtk.vtkAppendPolyData()

   vtpFile.SetFileName(fichier)
   vtpFile.SetDataMode(vtk.vtkXMLWriter.Binary)

   if 'meca_R' in fichier:
     nb = pushMecaR(AppendAll,dim,truc,kw)

   if nb :

     vtpFile.SetInputConnection(AppendAll.GetOutputPort())
     vtpFile.Write()

     time = TimeEvolution_GetTime()
     impr = '<DataSet timestep="%s" group="" part="0" file="%s"/>\n' % (time,'./'+os.path.basename(fichier))
     impr+= '</Collection>\n</VTKFile>'

     removeLines(fid)
     with open(fid,'a') as f:
       f.write(impr)

def writePolyrToVTK(fname, outline):

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writePolyrToVTK] Since vtk module is not available this function does nothing' )
      return

   vtuFile = vtk.vtkXMLUnstructuredGridWriter()
   vtuFile.SetFileName(fname)
   vtuFile.SetDataMode(vtk.vtkXMLWriter.Binary)

   # generate new unstructured grid
   ugrid = vtk.vtkUnstructuredGrid()
   nb = POLYR_GetNbPOLYR()

   if nb == 0:
     return

   points = vtk.vtkPoints()
   ugrid.SetPoints(points)

   cells  = ugrid.GetCells()

   # setting points
   POLYR_UpdatePostdata()
   for v in outline:
     points.InsertNextPoint(*v)

   offset   = 0
   for i_tact in range(nb):

     vertices = POLYR_GetPtrVertexRef(i_tact+1)
     connec = POLYR_GetPtrConnectivity(i_tact+1)
     for tri in connec:
       cell  = vtk.vtkTriangle()
       cellp = cell.GetPointIds()
       for i_node, i_t in enumerate(tri):
         cellp.SetId(i_node,offset+i_t-1)
       ugrid.InsertNextCell( cell.GetCellType(), cellp )
     offset = offset + vertices.shape[0]

   cellData  = ugrid.GetCellData()
   topo = POLYR_GetTopoData()

   nb_faces = topo.shape[0]
   for field in ["polyr_id", "topo_id", "face_id", "status"]:
     f = vtk.vtkIntArray()
     f.SetNumberOfComponents(1)
     f.SetNumberOfTuples(nb_faces)
     f.SetName(field)
     cellData.AddArray(f)

   p_id = cellData.GetArray('polyr_id')
   t_id = cellData.GetArray('topo_id')
   f_id = cellData.GetArray('face_id')
   stat = cellData.GetArray('status')
   for i in range(nb_faces):
     p_id.SetTuple1(i,topo[i,0])
     t_id.SetTuple1(i,topo[i,1])
     f_id.SetTuple1(i,topo[i,2])
     stat.SetTuple1(i,topo[i,3])

   vtuFile.SetInputData(ugrid)
   vtuFile.Write()

def get_new_cell(connec, csize):

    if csize == 2:
      elem = vtk.vtkLine()
    elif csize == 3:
      elem = vtk.vtkTriangle()
    elif csize == 4:
      elem = vtk.vtkQuad()
    else:
      raise ValueError( f'csize must be 2, 3 or 4 not {csize}' )

    for i, c in enumerate(connec):
      elem.GetPointIds().SetId(i, c)

    return elem

def writexSxxxToVTK(fpath):

   init_vtk()

   if ( not config.is_vtk_display ):
      print( '[WARNING:writexSxxxToVTK] Since vtk module is not available this function does nothing' )
      return

   ref_ifield_list = ["i_bdyty", "i_tacty", "i_xSxxx"]

   for fname in ['CSpxx', 'ASpxx', 'ALpxx', 'CLpxx']:
     if fname == 'CSpxx':
       ifield_list = ref_ifield_list+['quadrature',]
     elif fname == 'CLpxx':
       ifield_list = ref_ifield_list+['nb_nodes',]
     else:
       ifield_list = ref_ifield_list

     vtuFile = vtk.vtkXMLUnstructuredGridWriter()
     vtuFile.SetFileName(os.path.join(fpath,fname)+'.vtu')
     vtuFile.SetDataMode(vtk.vtkXMLWriter.Binary)

     # generate new unstructured grid
     ugrid = vtk.vtkUnstructuredGrid()

     points = vtk.vtkPoints()
     ugrid.SetPoints(points)

     cells  = ugrid.GetCells()

     xSxxx_connec = eval(fname+'_GetAllConnec()')
     idata, rdata = eval(fname+'_GetAllData()')
     if np.size(xSxxx_connec) == 0 :
       continue

     nb_xSxxx  = xSxxx_connec[0]
     #print( fname, ' nb xSxxx : ', nb_xSxxx )

     v_count = 0
     v2n = {}
     vertices  = None
     i_bdy_old = 0
     i_tac_old = 0

     idx = 1
     for i_s in range( nb_xSxxx ):

       i_bdy = idata[i_s,0]
       i_tac = idata[i_s,1]

       if i_bdy_old != i_bdy :
         v2n = {}
         i_bdy_old = i_bdy
         i_tac_old = 0
         vertices = mecaMAILx_GetCooref(int(i_bdy))
         if vertices.shape[1]==2:
           new_col  = np.zeros([vertices.shape[0],1])
           vertices = np.append(vertices, new_col ,axis=1)

       csize  = xSxxx_connec[idx]
       idx   += 1
       connec = xSxxx_connec[idx:idx+csize]

       # computing for current patch the vector to push vertices in
       if i_tac_old != i_tac :
         i_tac_old = i_tac
         push_in = np.linalg.norm(vertices[connec[1]-1]-vertices[connec[0]-1]) * 0.05
         push_in *= rdata[i_s,:3]

       for c in connec:
         # add vertices only if not added yet for current body
         v2n[c] = v_count
         coor = vertices[c-1,:] - push_in
         points.InsertNextPoint(*coor)
         v_count += 1
       nc = [v2n[c] for c in connec]
       elem = get_new_cell(nc, csize)
       ugrid.InsertNextCell(elem.GetCellType(),elem.GetPointIds())
       idx   += csize


     cellData  = ugrid.GetCellData()

     for field in ifield_list:
       f = vtk.vtkIntArray()
       f.SetNumberOfComponents(1)
       f.SetNumberOfTuples(nb_xSxxx)
       f.SetName(field)
       cellData.AddArray(f)

     norm = vtk.vtkDoubleArray()
     norm.SetNumberOfComponents(3)
     norm.SetNumberOfTuples(nb_xSxxx)
     norm.SetName('normal')
     cellData.AddArray(norm)

     fields = [cellData.GetArray(ifield) for ifield in ifield_list]
     for i_s in range( nb_xSxxx ):
         for i_f, field in enumerate(fields):
           field.SetTuple1(i_s,idata[i_s,i_f])
         norm.SetTuple3(i_s,*rdata[i_s,:3])

     vtuFile.SetInputData(ugrid)
     vtuFile.Write()


#######################################
# ecriture collections

def removeLines(fichier, ltd=2):
    # taken from stackoverflow :
    # smart way to remove last ltd lines from a file
    q = collections.deque()
    ltd = 2
    for line in fileinput.input(fichier, inplace=True):
      q.append(line)
      if ltd == 0:
        print(q.popleft(), end='')
      else:
        ltd -= 1
    q.clear()

def startCollection(fichier,restart=0):
  if restart > 0 and os.path.isfile(fichier) :
    # remove all lines from file after finding pattern
    # (https://stackoverflow.com/questions/4710067/how-to-delete-a-specific-line-in-a-file)
    removeLines(fichier)
    reg = re.compile(r'file="\./\w+_(?P<step>\d+).vtu"/>')
    with open(fichier, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            se = reg.search(line)
            step = int(se['step']) if se is not None else 0
            if step > restart:
                break
            else:
                f.write(line)
        f.truncate()
        closing = '</Collection>\n</VTKFile>'
        f.write(closing)
  else :
    with open(fichier,'w') as fid:
        header= '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n'
        fid.write(header)
        closing = '</Collection>\n</VTKFile>'
        fid.write(closing)
  return fichier

