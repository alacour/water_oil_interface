import numpy as np
import matplotlib.pyplot as plt
import signac
import freud
from scipy.stats import uniform
from scipy.optimize import curve_fit
from scipy.special import erf
from joblib import Parallel, delayed
from skimage import measure
import pandas as pd
import mdtraj

def instantaneous_interface(gridmul):
    #Loading Data
    traj = mdtraj.load_dcd("../../combined_sampled_trajectory.dcd", "../../begin.pdb") 

    poses = []
    boxes = []
    for i in range(len(traj)):
        positions = traj.xyz[i] * 10
        top = traj.topology
        residues = traj.topology.residues
        atoms = traj.topology.atoms
        box = traj.unitcell_lengths[i] * 10
        
        poses.append(positions)
        
        boxes.append(box)

    print(len(poses))
    print(residues)

    type_list = []
    for res in residues:
        res = str(res)
        #print(res)
        if res[:3] == 'OHO':
            type_list.append(0)
            type_list.append(1)
            type_list.append(1)
            
        if res[:3] == 'HEX':
            for i in range(3):
                type_list.append(4)
                
            type_list.append(2)


            for i in range(4):
                for j in range(2):
                    type_list.append(5)
                type_list.append(3)
                
        if res[:3] == 'HEX':
            for i in range(3):
                type_list.append(4)
                
            type_list.append(2)            


    type_list = np.array(type_list)

    #Sorting into Opos and Hpos centered around z = 0
    Oposes = []
    Hposes = []
    start = 0
    for i in range(0, len(poses), 1): 

        fbox = freud.box.Box.from_box(boxes[i])
        pos = poses[i]

        Opos =  pos[type_list == 0]
        Hpos =  pos[type_list == 1]
        
        Hpos = fbox.wrap(Hpos + np.array([0, 0, 10]))
        Opos = fbox.wrap(Opos + np.array([0, 0, 10]))
        C1pos = pos[type_list == 2]
        C1pos = pos[type_list == 3]
        H1pos = pos[type_list == 4]
        H2pos = pos[type_list == 5]
        
        
        Hpos = fbox.wrap(Hpos - [0, 0, np.average(Opos[:,2])])
        Opos = fbox.wrap(Opos - [0, 0, np.average(Opos[:,2])])

        Oposes.append(Opos)
        Hposes.append(Hpos)
        
    Oposes = np.array(Oposes)
    Hposes = np.array(Hposes)

    # Computing GDS
    water_function = lambda x, xo, scale, height: height*(erf(scale*(x - xo))/2 + 0.5)


    boxW = box[0]**2 #Box area
    start = 100 #Earliest Frame to include in data
    tot = np.concatenate(Oposes[start:])
    un, co = np.unique(np.round(tot[:,2], 1), return_counts=True)
    dens = co/(len(Oposes) - start)/boxW/0.1/(1/18.02 * 6.022 * 10**23 / 10**24)
          

    print(len(Oposes))
    rwater_var, rwater_cov = curve_fit(water_function, 
                                       un[un > 0], 
                                       dens[un > 0], 
                                       [18, 1, 1],
                                       maxfev = int (2e5))

    xx = np.linspace(rwater_var[0] - 8, rwater_var[0]+8)
    yy = water_function(xx, *rwater_var)

    print(rwater_var)
    print(box)

    #Setting up Grid for Instantaneous Interface Calc.

    gridlenx = 36*gridmul
    dx = 1/gridlenx
    xs = np.linspace(dx/2, 1 - dx/2, gridlenx)
    ys = np.copy(xs)

    gridlenz = 16*gridmul
    dz = 1/gridlenz
    zs = np.linspace(dz/2, 1 - dz/2, gridlenz)

    grid = []
    for x in xs:
        #print(x)
        for y in ys:
            for z in zs:
                grid.append([x, y, z])
                
    grid = np.array(grid)
    print(len(grid))
    print(grid)


    #Instantaneous Interface Calc.


    gds = rwater_var[0]
    extantz = 16 # Length of Grid in Z direction
    chi = 2.4 # Gaussian Width to Use
    rmax = chi*3.0 # Where to Truncate Gaussian
    bdis = []  # Oxygen distances
    Hbdis = []  # Hydrogen distances


    def instantaneous_surface(k):
        Opos = np.copy(Oposes[k])
        Hpos = np.copy(Hposes[k])
        box = np.copy(boxes[k])
        
        # Setting up grigd
        mgrid = np.copy(grid)
        mgrid[:,:2] = mgrid[:,:2]*box[:2] - box[:2]/2
        
        mgrid[:,2] = mgrid[:,2] * extantz - extantz/2 + gds
        
        
        # Computing Density on Grid
        tOpos = Opos[Opos[:,2] > 0]
        fbox = freud.box.Box.from_box(box)
        aq = freud.locality.AABBQuery(fbox, tOpos)
        nlist = aq.query(mgrid, {'r_max': rmax}).toNeighborList()   
        dgs = np.array([dx*box[0], dx*box[1], dz*extantz])
        dis = np.linalg.norm(fbox.wrap(tOpos[nlist.point_indices] - mgrid[nlist.query_point_indices]), axis=1)
        gauss = 1/(2*np.pi*chi**2)**(3/2)*np.exp(-0.5*(dis/chi)**2)
        gauss = gauss - 1/(2*np.pi*chi**2)**(3/2)*np.exp(-0.5*(rmax/chi)**2)
        
        un, co = np.unique(nlist.query_point_indices, return_counts=True)
        dens = []
        it = 0
        total = np.arange(0, len(mgrid))
        dens = np.zeros(len(grid))
        i = 0
        for u in un:
            et = it + co[i]
            wh = np.arange(it, et)
            dens[u] = np.sum(gauss[wh])
            it = et
            i += 1

        tgrid = np.reshape(dens, [gridlenx, gridlenx, gridlenz])

        # Finding Interface   
        dens = np.array(dens)
        verts, faces, normals, values = measure.marching_cubes(tgrid, 0.016, spacing=(dgs[0], dgs[1], dgs[2]))
        verts[:,2] =  verts[:,2] - extantz/2 + gds + dgs[2]/2
        verts[:,:2] =  verts[:,:2] - box[:2]/2 + dgs[:2]/2

        # Finding how close particles are to Interface
        nneighbors = 1
        tOpos = np.copy(Opos[Opos[:,2] > 0])
        gbox = box * np.array([1, 1, 4])
        fbox = freud.box.Box.from_box(gbox)

        aq = freud.locality.AABBQuery(fbox, verts)
        nlist = aq.query(tOpos, {'num_neighbors': nneighbors}).toNeighborList()   
        #print('here1')
        directs = fbox.wrap(verts[nlist.point_indices] - tOpos[nlist.query_point_indices])
        dis = np.linalg.norm(directs, axis=1)
        normed = (directs.T / dis).T
        signs = np.sign(normed[:,2])

        dis = np.sum(directs*normed, axis=1) * signs
        bdis.append(dis)  
        
        
        tHpos = np.copy(Hpos[Hpos[:,2] > 0])
        gbox = box * np.array([1, 1, 4])
        fbox = freud.box.Box.from_box(gbox)

        aq = freud.locality.AABBQuery(fbox, verts)
        nlist = aq.query(tHpos, {'num_neighbors': nneighbors}).toNeighborList()   
        directs = fbox.wrap(verts[nlist.point_indices] - tHpos[nlist.query_point_indices])
        dis = np.linalg.norm(directs, axis=1)
        normed = (directs.T / dis).T
        signs = np.sign(normed[:,2])

        dis = np.sum(directs*normed, axis=1) * signs
        Hbdis.append(dis)
        
        return bdis, Hbdis
        
    
    
    processors = 10 
    disses = Parallel(n_jobs=processors, verbose=10)(delayed(instantaneous_surface)(k) for k in range(0, len(Oposes), 1))

    disses = np.array(disses, dtype='object')

    Odisses = np.concatenate(disses[:,0])
    Hdisses = np.concatenate(disses[:,1])

    np.save("Odisses.npy", Odisses)
    np.save("Hdisses.npy", Hdisses)




