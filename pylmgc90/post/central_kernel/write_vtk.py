# needed to read again :
# 
# https://docs.paraview.org/en/latest/UsersGuide/understandingData.html
# https://kitware.github.io/vtk-examples/site/Python/GeometricObjects/Polygon/
# https://kitware.github.io/vtk-examples/site/VTKFileFormats/

from pathlib import Path

import numpy as np

import vtk
from ...chipy.vtk_display import removeLines

def push_ck(f2f, ck, ck_list):

    for i_f2f, data in enumerate(f2f):

        coor, reac, rn = data

        nb_v = coor.shape[0]

        # each polydata as at least 1 face (the original polygon)
        # if ck is not None a second face (the central kernel)
        # if xc is not None a third face (the null moment force point)
        nb_poly = 1
        nb_v_ck = 0
        nb_v_xc = 0

        ck_coor, sigma, xc_coor, is_in = ck[i_f2f]
        nb_poly += 3
        nb_v_ck  = ck_coor.shape[0]
        nb_poly += 1
        nb_v_xc  = 1

        nb_total_v = nb_v+nb_v_ck+nb_v_xc

        pdata = vtk.vtkPolyData()
        pdata.Allocate(nb_poly)
    
        # allocate all the points
        pts   = vtk.vtkPoints()
        pts.SetNumberOfPoints(nb_total_v)

        faces = vtk.vtkCellArray()
        verts = vtk.vtkCellArray()

        # map the f2f points to a polygon
        polyg = vtk.vtkPolygon()
        polyg.GetPointIds().SetNumberOfIds(nb_v)
        for i_coor in range(nb_v):
            i_point = i_coor
            c = coor[i_coor]
            pts.SetPoint(i_point, c[0], c[1], c[2])
            polyg.GetPointIds().SetId(i_coor, i_point)
        faces.InsertNextCell(polyg)

        # map the central kernel points to a polygon
        ck_face = vtk.vtkPolygon()
        ck_face.GetPointIds().SetNumberOfIds(nb_v_ck)
        for i_coor in range(nb_v_ck):
            i_point = nb_v+i_coor
            c = ck_coor[i_coor]
            pts.SetPoint(i_point, c[0], c[1], c[2])
            ck_face.GetPointIds().SetId(i_coor, i_point)
        faces.InsertNextCell(ck_face)
    
        xc_vert = vtk.vtkVertex()
        xc_vert.GetPointIds().SetNumberOfIds(nb_v_xc)
        for i_coor in range(nb_v_xc):
            i_point = nb_v+nb_v_ck+i_coor
            c = xc_coor
            pts.SetPoint(i_point, c[0], c[1], c[2])
            xc_vert.GetPointIds().SetId(i_coor, i_point)
        verts.InsertNextCell(xc_vert)

        # link container of points/faces to polydata
        pdata.SetPoints(pts)
        pdata.SetPolys(faces)
        pdata.SetVerts(verts)

        # adding field to cells/points of polydata
        cellData  = pdata.GetCellData()
        pointData = pdata.GetPointData()

        # string field
        ftype = vtk.vtkStringArray()
        ftype.SetNumberOfComponents(1)

        # integer scalar field for f2f id
        ids = vtk.vtkIntArray()
        ids.SetNumberOfComponents(1)
        ids.SetNumberOfTuples(nb_poly)
        ids.SetName("Ids")
        for i in range(nb_poly):
          ids.SetTuple1(i, i_f2f)
        cellData.AddArray(ids)

        # integer scalar field for cell type id
        cell_type = vtk.vtkIntArray()
        cell_type.SetNumberOfComponents(1)
        cell_type.SetNumberOfTuples(nb_poly)
        cell_type.SetName("Type")
        idx = 0
        cell_type.SetValue(idx, 0)
        idx += 1
        if nb_v_ck:
          cell_type.SetValue(idx, 1)
          idx += 1
        if nb_v_xc:
          cell_type.SetValue(idx, 2)
          idx += 1
        cellData.AddArray(cell_type)

        # real vector field
        reac_field = vtk.vtkFloatArray()
        reac_field.SetNumberOfComponents(3)
        reac_field.SetNumberOfTuples(nb_total_v)
        reac_field.SetName("Reac")
        for i_point in range(nb_v):
            r = reac[i_point]
            reac_field.SetTuple3(i_point, r[0], r[1], r[2])
        pointData.AddArray(reac_field)

        sigma_field = vtk.vtkFloatArray()
        sigma_field.SetNumberOfComponents(1)
        sigma_field.SetNumberOfTuples(nb_poly)
        sigma_field.SetName("Sigma_n")
        for i in range(nb_poly):
          sigma_field.SetTuple1(i, sigma)
        cellData.AddArray(sigma_field)

        r = np.sum(reac, axis=0)
        reac_field.SetTuple3(nb_v+nb_v_ck, r[0], r[1], r[2])
    
        # integer scalar field for status of xc (in or out of ck)
        cell_status = vtk.vtkIntArray()
        cell_status.SetNumberOfComponents(1)
        cell_status.SetNumberOfTuples(nb_poly)
        cell_status.SetName("Status")
        cell_status.SetValue(1, int(False) )
        cell_status.SetValue(0, int(not is_in))
        cell_status.SetValue(2, int(not is_in))
        cellData.AddArray(cell_status)

    
        ck_list.AddInputData( pdata )


def write_vtk(time, filename, fid, f2f, ck):

    # create file generator
    vtk_file = vtk.vtkXMLPolyDataWriter()
    vtk_file.SetFileName(str(filename))
    
    # a container of polydata
    ck_list = vtk.vtkAppendPolyData()
    
    # creating a polydata for each f2f
    push_ck( f2f, ck, ck_list)

    vtk_file.SetInputConnection(ck_list.GetOutputPort())
    vtk_file.Write()

    impr = '<DataSet timestep="%s" group="" part="0" file="%s"/>\n' % (time,filename.name)
    impr+= '</Collection>\n</VTKFile>'

    removeLines(str(fid))
    with open(fid,'a') as f:
      f.write(impr)

