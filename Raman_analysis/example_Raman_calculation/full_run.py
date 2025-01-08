from necessary_functions import *

def compute_distribution(ii, kstretch, kb1, kb2, kbendstd, nrandos, end, leng):

    hprojs = np.load('../../amoeba_pair_fields.npy', allow_pickle=True)  # Fields experienced by Hs
    interface_dis = np.load('../../Odisses.npy', allow_pickle=True) # Distance to WCI
    oilangles  = np.load('../../amoeba_angles.npy', allow_pickle=True)  # Hydrogen bonding angles


    Oposes = oilangles[ii][:,0]  # Zposition of oxygens
    projs = hprojs[ii] 
    angles = oilangles[ii][:,1:]
   
    alli = interface_dis[ii]
    projs = projs[Oposes > 0]  #  Only computed WCI distances for waters with Zpositions > 0
    angles = angles[Oposes > 0]

    lenf = len(projs)
    print(len(alli), len(projs), len(angles), len(interface_dis)) 
    randos = np.argsort(np.random.uniform(0, 1, lenf))[:nrandos] # Random subset
    alli = alli[randos]
    projs = projs[randos]
    angles = angles[randos]

    print(len(alli), len(projs))

       
    bound = 90

    cond =  (alli > end) *  (alli < end + leng) # only getting waters within a certain distance from the WCI
    projs = projs[cond]*14.4 #Converting to V/A
    angles = angles[cond]


    cond = (angles[:,0] > bound) * (angles[:,1] > bound) # Making Sure I only get waters with both  OH's bonded
    projs = projs[cond]
    angles = angles[cond]


    processors = 20
    theta0 = 109
    nbasis = 6
    m1 = 15.999
    m2 = 1.007
    m3 = 1.007
    hbar1 = 6.5821 * 10**-16
    hr1 = hbar1*2*np.pi
    hbar2 = hbar1 * 1.602*10**-19 * 6.022*10**26 * 10**20
    h2 = hbar2*2*np.pi
    h12 = hbar1 * hbar2
    mul1 = 50
    mul2 = 100
    mul3 = 100

    fields = np.copy(projs)
    bendfields = np.copy(projs) #Identifical to fields
    muls = np.random.normal(1.0, kbendstd, nrandos)  # multipliers for bending energy
    np.save('muls', muls) 
    np.save('used', randos)
    eig_contain = []
    allumasses = []
    harvecs = []
    minposes = []
    mfields = []
    print("here 1")

    for i,field in enumerate(fields):
        g = eigenvalues(field, kstretch, kb1*muls[i], kb2*muls[i], bendfields[i])
        if np.sum(np.isnan(g[2])) < 1:
            allumasses.append(g[0])
            eig_contain.append(g[1])
            harvecs.append(g[2])
            minposes.append(g[3])
            mfields.append([field, bendfields[i]])
            print(i)
        else:
            print("fail here", g[2])
   
    eig_contain = np.asarray(eig_contain)   
    np.save("harvecs", harvecs)
    np.save("umasses", allumasses)
    np.save("minposes", minposes)
    np.save("fields", mfields)

    basis0 = 0.0
    basisL = 1.1
    xmin = basis0 - basisL/2
    xmax = basis0 + basisL/2
    xx = np.linspace(xmin, xmax, 20)
    stepsize = xx[1] - xx[0]
    print(stepsize)
    vol = (basisL + stepsize)**3
    xxgrid = np.array([np.repeat(xx, len(xx)**2),
                       np.repeat(xx.tolist()*len(xx), len(xx)),
                       xx.tolist()*len(xx)**2]).T

    energies = []
    vecadds = []
    egrids = []
    polgrids = []
    t =time()
    sleep(10)
    def fill_grids(j):
        vectors = harvecs[j]
        poses = minposes[j] 
        poses = np.array(len(xx)**3*[poses])
        field = mfields[j][0]
        bendfield = mfields[j][1]
        field1 = field[0]
        field2 = field[1]
        bendfield1 = bendfield[0]
        bendfield2 = bendfield[1]
        
        mul = muls[j]
        umasses = []
        for i in range(3):
            vecadd = np.reshape(np.copy(vectors[:,-3+i]), [3, 3])
            nkeep = np.copy(vecadd)
            #print(vecadd)
            #print(np.linalg.norm(vecadd))
            vecadd[0] = vecadd[0]/m1**(1/2)
            vecadd[1] = vecadd[1]/m2**(1/2)
            vecadd[2] = vecadd[2]/m3**(1/2)


            vecadd = vecadd / np.linalg.norm(vecadd)
            vecadds.append(vecadd)
            dkeep = np.copy(nkeep)


            dkeep[0] = dkeep[0]**2/m1
            dkeep[1] = dkeep[1]**2/m2
            dkeep[2] = dkeep[2]**2/m3



            umasses.append(1/np.sum(dkeep))
            
        vecadd1 = np.array(len(xx)**3*[vecadds[0]])
        vecadd2 = np.array(len(xx)**3*[vecadds[1]])
        vecadd3 = np.array(len(xx)**3*[vecadds[2]])


        posgrid = poses + (((xxgrid[:,0]*vecadd1.T).T) + 
                           (xxgrid[:,1]*vecadd2.T).T + 
                           (xxgrid[:,2]*vecadd3.T).T)
        cgrid = extract_coordinates(posgrid)
        egrid = fastonebody(cgrid, field1, field2, bendfield1, bendfield2, kstretch, kb1*mul, kb2*mul)
        polgrid = []
        for pos in posgrid:
            pol = onepol(pos)
            polgrid.append(pol)
            
    #    print(j)
    #    print(time()-t)

        return egrid, polgrid, umasses

    grids = Parallel(n_jobs=processors, verbose=10)(delayed(fill_grids)(i) for i in range(len(harvecs[:])))
    sleep(20)

    for i in range(len(grids)):
        egrids.append(grids[i][0])
        polgrids.append(grids[i][1])
        allumasses.append(grids[i][2])

    psis = []
    d2d2xs = []
    d2d2ys = []
    d2d2zs = []



    for i in range(0, nbasis):
        for k in range(0, nbasis):
            for m in range(0, nbasis):
                #psis.append(basis3(i, k, m, basisL, basis0, xxgrid[:,0], xxgrid[:,1], xxgrid[:,2]))
                
                psi1, d2d2x, d2d2y, d2d2z = numerical_derivative(i, k, m, 
                                                        xxgrid[:,0], 
                                                        xxgrid[:,1], 
                                                        xxgrid[:,2], 
                                                        h12, 
                                                        mul1,
                                                        mul2,
                                                        mul3) 
                psis.append(psi1)
                d2d2xs.append(d2d2x)
                d2d2ys.append(d2d2y)
                d2d2zs.append(d2d2z)
                
                

    psis = np.array(psis)
    d2d2xs = np.asarray(d2d2xs)
    d2d2ys = np.asarray(d2d2ys)
    d2d2zs = np.asarray(d2d2zs)


    print(len(psis))
    vol = (basisL + stepsize)**3

    def compute_hamiltonians(kt):
        umasses = allumasses[kt]
        egrid = egrids[kt]
        es = []

       
        for it, psi1 in enumerate(psis[:]):
            d2d2x = d2d2xs[it]
            d2d2y = d2d2ys[it]
            d2d2z = d2d2zs[it]
            jt = 0
            energies = []


            for jt, psi2 in enumerate(psis[:]):

                dx = d2d2x/umasses[0]
                dy = d2d2y/umasses[1]
                dz = d2d2z/umasses[2]
                    
                inte1 = np.average(psi2*(dx + dy + dz))*vol
                inte2 = np.average(psi2*psi1*egrid)*vol
                energies.append(-inte1+inte2)
            es.append(energies)
        return es

    ematrices = Parallel(n_jobs=processors, verbose=10)(delayed(compute_hamiltonians)(i) for i in range(len(harvecs[:])))
    sleep(20)


    reseigs = []
    resvecs = []
    print(len(ematrices))
    for i in range(len(egrids)):
        ees = ematrices[i]
        eigs, vecs = np.linalg.eigh(ees)
        reseigs.append(eigs)
        resvecs.append(vecs)
        gap = eigs[1] - eigs[0]
        wavenumber = gap*8065.544
        print(mfields[i])
        print(wavenumber)

        gap = eigs[2] - eigs[0]
        wavenumber = gap*8065.544
        print(wavenumber)
        gap = eigs[3] - eigs[0]
        wavenumber = gap*8065.544
        print(wavenumber)

        gap = eigs[4] - eigs[0]
        wavenumber = gap*8065.544
        print(wavenumber)
        print(len(eigs))
        print(len(psis))
        print(i)

    reseigs = np.asarray(reseigs)
    resvecs = np.asarray(resvecs)
    np.save('qeigs.npy', reseigs) 
    reseigs = np.asarray(reseigs)
    resvecs = np.asarray(resvecs)
    freqs = (reseigs[:,1:].T - reseigs[:,0]).T
    freqs = freqs[:,:4]*8065.544
    np.save('qfreqs.npy', freqs)



    relvecs = resvecs[:,:,0:5]

    t = time()
    activities = []

    for it,vecs in enumerate(resvecs):
        polgrid = np.asarray(polgrids[it])
        psi0 = np.sum(psis.T*vecs[:,0], axis=1).T
        activity = [[],[]]
        for i in range(4):
            psij = np.sum(psis.T*vecs[:,i+1], axis=1).T
            pol1 = np.average(polgrid.T*psi0*psij, axis=2).T*vol
            sym = np.sum(np.diag(pol1))**2
            beta = np.copy(pol1)
            beta_trace = np.sum(np.diag(beta))
            beta -= beta_trace/3 * np.diag([1, 1, 1])
            anti = np.sum(np.diag(np.matmul(beta, beta)))
            activity[0].append(sym)
            activity[1].append(anti)
            
        activities.append(np.concatenate(activity))

        print(it)

    activities = np.array(activities)
    np.save('activities.npy', activities)
    return 0 


