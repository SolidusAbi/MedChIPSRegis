# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 11:53:04 2018

@author: MariaArmas
"""

import numpy as np

#def write_unstructured_file(Ym, r_values, g_values, b_values):
def write_unstructured_file(file_name,Ym,r_values):
    r_values = r_values.reshape(25,1)
    g_values = r_values
    b_values = r_values
    n_points = Ym.shape[0]
    height = 5 #Number of rows
    width = 5 #Number of columns
    n_cells = (height-1)*(width-1)
    n_cell_values = n_cells*5 
    alpha = 0.5 #opacity of the rgb values
      
    #Create a DATAFILE
    fd = open("{}".format(file_name),"w") #vtk unstructured_grid_prueba_source.vtk
    fd.write("# vtk DataFile Version 4.2\n")
    fd.write("vtk output\n")         
    fd.write("ASCII \n\n")
    fd.write("DATASET UNSTRUCTURED_GRID \n")         
    fd.write("POINTS {} float \n".format(n_points))   
    
    points = np.arange(0,n_points)
    p = (np.arange(1,width+1))*(height)    
    p = p-1 
    p = np.append(p,np.arange(n_points-width,n_points))
    points_cell = np.delete(points,p)
    
    for i in range(n_points):
        fd.write("{} {} 0\n".format(Ym[i][0], Ym[i][1]))
    
    fd.write("\nCELLS {} {}\n".format(n_cells, n_cell_values))
    
    for j in (points_cell):
        fd.write("4 {} {} {} {}\n".format(points[j], points[j+1], points[j+height+1], points[j+height]))

    fd.write("\nCELL_TYPES {}\n".format(n_cells))
    
    for k in range(n_cells):
        fd.write("9 \n")
        
    fd.write("\nPOINT_DATA {}\n".format(n_points))
    fd.write("SCALARS scalars float 4\n")
    fd.write("LOOKUP_TABLE my_table \n")
    for l in range(n_points):
        fd.write("{} {} {} {}\n".format(r_values[l][0], g_values[l][0],
                 b_values[l][0], alpha))
    
    fd.close()       
           


