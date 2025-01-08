import numpy as np
from scipy.optimize import curve_fit, minimize
from joblib import dump, load
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
from time import time,sleep
from scipy.spatial.transform import Rotation as R
from scipy.special import factorial, hermite
from joblib import Parallel, delayed

shifts = np.load('one_body_shifts.npy', allow_pickle=True)
krr = load('one_body_ridge.joblib')
poly = PolynomialFeatures(12)
pre = lambda x: krr.predict(poly.fit_transform(x))

def onebody(p):
    p = p
    d1 = p[1] - p[0]
    d2 = p[2] - p[0]
    stretch1 = np.linalg.norm(d1)
    stretch2 = np.linalg.norm(d2)
    pr = np.sum(d1*d2)/stretch1/stretch2
    angle = np.arccos(np.round(pr, 8)) *180/np.pi

    ar = ((np.array([[stretch1, stretch2, angle]]) - shifts)/shifts).astype('float64')
    return krr.predict(poly.fit_transform(ar))[0]


def fastonebody(array, field1, field2, bendfield1, bendfield2, kstretch, kb1, kb2):
    inputs = ((array - shifts) / shifts).astype('float64')
#    print(field1, bendfield1)
    fieldenergy = (-kstretch*(array[:,0]*field1 + array[:,1]*field2) +  -kstretch*((array[:,0] - 0.958929)**2*field1 + (array[:,1] - 0.958929)**2*field2)*0.79/0.37/2
                   + (bendfield1 + bendfield2)*(kb1*(array[:,2] - 104.3636) + kb2*(array[:,2] - 104.3636)**2))
    return fieldenergy + krr.predict(poly.fit_transform(inputs))

krrpol = load('polarizability_krr.joblib')

def prepol(x):
    x[:,:2] = np.flip(np.sort(x[:,:2], axis=1), axis=1)
    return krrpol.predict(polypol.fit_transform(x))

polypol = PolynomialFeatures(5)


def onepol(p):
    vec_aim = np.array([1.0, 0, 0])
    d1 = p[1] - p[0]
    d2 = p[2] - p[0]
    s1 = np.linalg.norm(d1)
    s2 = np.linalg.norm(d2)
 
    npos = np.array([[0, 0, 0], d1, d2])         
    cross = np.cross(d1, d2)
    cross = cross / np.linalg.norm(cross)

   
    
    add = np.array([0, 0, 1])
    bisect = add + cross   
    
    if (np.linalg.norm(bisect) > 1e-8) * (np.abs(np.linalg.norm(bisect) - 2) > 1e-8):
        bisect = bisect / np.linalg.norm(bisect)
        qcross = R.from_quat([bisect[0], bisect[1], bisect[2], 0])
        crossmat = qcross.as_matrix()
        npos = qcross.apply(npos)
    else:
        qcross = R.from_quat([0, 0, 0, 1])
        crossmat = qcross.as_matrix()
        npos = np.copy(npos)

    
    orientation = (npos[1]/s1 + npos[2]/s2)
    orientation = orientation / np.linalg.norm(orientation)
    pr = np.sum(vec_aim * orientation)
    angle = np.arccos(np.round(pr, 8))

    sign = np.sign(np.cross(vec_aim, orientation)[2])
    angle = -angle*sign
    
    if abs(angle) > 1e-8:
        qtwist = R.from_quat([0, 0, np.sin(angle/2), np.cos(angle/2)])
        revtwist = R.from_quat([0, 0, np.sin(-angle/2), np.cos(-angle/2)])
        revtwistmat = revtwist.as_matrix()
        npos = qtwist.apply(npos)

    else:
        qtwist = R.from_quat([0, 0, 0, 1])
        revtwist = qtwist
        twistmat = qtwist.as_matrix()
        revtwistmat = revtwist.as_matrix()
        npos = qtwist.apply(npos)
  
    final_flip = 1
    if npos[2,0] > npos[1,0]:
        final_flip = -1
    
    pr = np.sum(npos[1]*npos[2])/s1/s2
    angle = np.arccos(np.round(pr, 8))*180/np.pi

    inarray = (np.array([[s1, s2, angle]]))# - shifts)/shifts
    
    perpol =  prepol(inarray)[0]
    perpol[[1, 3]] = final_flip*perpol[[1, 3]]
    perpol = np.reshape(perpol, [3, 3])
    inverse_mat = np.matmul(crossmat, revtwistmat)
    pol = np.matmul(np.matmul(inverse_mat, perpol), np.linalg.inv(inverse_mat))

    
    return pol


#g = np.array([[0, 0, 0],
#              [0.961, 0, 0],
#              [-0.23, 0.93, 0]])*1.2


#print(onepol(g))


def minima_search(efield1, efield2, bendfield1, bendfield2, min1, min2, min3, kstretch, kb1, kb2):
    
    def elambda(ass):
        a1, a2, a3 = ass
        evaluates = (ass - shifts)/shifts
        unperturbed = pre(np.array([evaluates]))
        perturbed = unperturbed  + stretch_energy(a1, a2, efield1, efield2, kstretch)
        perturbed = perturbed + angle_energy(a3, efield1, efield2, bendfield1, bendfield2, kb1, kb2)

        return perturbed
        
    mini = minimize(elambda, 
                    x0 = [min1, min2, min3], 
                    method = 'L-BFGS-B', 
                    options = {"eps":1e-7, "maxiter":int(1e4)}, 
                    tol=1e-20)
    
    
    #print(mini)
    #print(efield1, efield2)
    return mini.x

ran = [2.0]

#f = minima_search(ran[0], ran[0], 1, 1, 100, 0.5, 0, 109)
#print(f)

def stretch_energy(stretch1, stretch2, efield1, efield2, kstretch):
    stretch1 = stretch1 - 0.958929
    stretch2 = stretch2 - 0.958929
    return   -kstretch*(efield1*stretch1 + efield2*stretch2) +  -kstretch*(efield1*stretch1**2 + efield2*stretch2**2)*0.79/0.37/2

def angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2):
    angle = angle - 104.3636
    return   (bendfield1 + bendfield2)*(kb1*angle + kb2*angle**2)

def construct_mat(Opos, H1pos, H2pos, efield1, efield2, bendfield1, bendfield2, kstretch, kb1, kb2):
    mat = []
    allvar = np.concatenate([Opos, H1pos, H2pos])
    de = 1e-4

    for i in range(9):
        tmat = []
        for j in range(9):
            tvar = np.copy(allvar)
            tvar[i] += de
            tvar[j] += de
            #print(tvar)
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8))*180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
            eff = pre(ar)[0] + bendterm + angleterm
            
            tvar[j] -= 2*de
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8))*180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
            efb = pre(ar)[0] + bendterm + angleterm
            
            
            tvar[i] -= 2*de
            tvar[j] += 2*de
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8)) *180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
            ebf = pre(ar)[0] + bendterm + angleterm
            
            tvar[j] -= 2*de
            d1 = tvar[3:6] - tvar[:3]
            d2 = tvar[6:] - tvar[:3]
            stretch1 = np.linalg.norm(d1)
            stretch2 = np.linalg.norm(d2)
            pr = np.sum(d1*d2)/stretch1/stretch2
            angle = np.arccos(np.round(pr, 8)) *180/np.pi

            ar = (np.array([[stretch1, stretch2, angle]]) - shifts)/shifts
            bendterm = stretch_energy(stretch1, stretch2, efield1, efield2, kstretch)
            angleterm = angle_energy(angle, efield1, efield2, bendfield1, bendfield2, kb1, kb2)
            ebb = pre(ar)[0] + bendterm + angleterm


            dde = (eff + ebb - efb - ebf)/4/de**2

            tmat.append(dde)


        mat.append(tmat)


    mat = np.asarray(mat)
    return mat

def reduced_mass(mat, m1, m2, m3):
    tmat = np.copy(mat)
    for i in range(9):
        for j in range(9):
            if i < 3:
                mass1 = m1

            elif i - 3 < 3:
                mass1 = m2
                
            else:
                mass1 = m3

            if j < 3:
                mass2 = m1

            elif j - 3 < 3:
                mass2 = m2
            else:
                mass2 = m3
            tmat[i,j] = tmat[i,j]  / mass1**(1/2) / mass2**(1/2)
            
            
    return tmat

def eigenvalues(field, kstretch, kb1, kb2,  bendfield):
    m1 = 15.999
    m2 = 1.007
    m3 = 1.007
    conv = (1*2*np.pi*3*10**10)**(2)*1.66054e-27/10**20*6.242*10**18
    field1 = field[0]
    field2 = field[1]
    bendfield1 = bendfield[0]
    bendfield2 = bendfield[0]

    f = minima_search(field1, field2, bendfield1, bendfield2, 1, 1, 100, kstretch, kb1, kb2)
    stretch1, stretch2, thetastart = f
    Opos = np.array([0, 0, 0])
    H1pos = np.array([1, 0, 0])*stretch1
    H2pos = np.array([np.cos(thetastart/180*np.pi), np.sin(thetastart/180*np.pi), 0])*stretch2
    mat = construct_mat(Opos, H1pos, H2pos, field1, field2, bendfield1, bendfield2, kstretch, kb1, kb2)
    tmat = reduced_mass(mat, m1, m2, m3)
    eigs, vectors = np.linalg.eigh(tmat)
    freq = (eigs/conv)[6:]**(1/2)
    
    moves = np.reshape(vectors[:,6:], [3, 3, 3])

    tmoves = np.copy(moves)
    umasses = []
    for i in range(3):
        trans = moves[:,:,i] / (np.array([[m1, m2, m3]]).T**(1/2))
        trans = trans / np.linalg.norm(trans)
        tmoves[:,:,i] = trans
        
        dkeep = np.copy(moves[:,:,i])
        dkeep[0] = dkeep[0]**2/m1
        dkeep[1] = dkeep[1]**2/m2
        dkeep[2] = dkeep[2]**2/m3


        umasses.append(1/np.sum(dkeep))
        
    eig_contain =([field1, field2, bendfield1, bendfield2,
                        freq[0], freq[1], freq[2],
                        ])
    minpos = np.array([Opos, H1pos, H2pos])

    return umasses, eig_contain, vectors, minpos



def extract_coordinates(poses):
    s1 = poses[:,1] - poses[:,0]
    s2 = poses[:,2] - poses[:,0]
    stretch1 = np.linalg.norm(s1, axis=1)
    stretch2 = np.linalg.norm(s2, axis=1)
    pr = np.sum(s1*s2, axis=1)/stretch1/stretch2
    angle = np.arccos(np.round(pr, 8))*180/np.pi
    return np.array([stretch1, stretch2, angle]).T


def basis3(n1, n2, n3, x, y, z, mul1, mul2, mul3):
    
    f = ((mul1*mul2*mul3/np.pi**3)**(1/4))
    
    f = f/np.sqrt(2**n1*factorial(n1)*
                  2**n2*factorial(n2)*
                  2**n3*factorial(n3))
    
    f = f*((np.exp(-mul1/2*x**2))*
           (np.exp(-mul2/2*y**2))*
           (np.exp(-mul3/2*z**2)))
    
    f = f*(hermite(n1)(np.sqrt(mul1)*(x))*
           hermite(n2)(np.sqrt(mul2)*(y))*
           hermite(n3)(np.sqrt(mul3)*(z)))
    
    
    return f


def numerical_derivative(i, j, k, x, y, z, h12,  mul1, mul2, mul3):
    
    ddd = 1e-5
    psi = basis3(i, j, k, x, y, z, mul1, mul2, mul3)



    fpsi = basis3(i, j, k,  x + ddd, y, z,  mul1, mul2, mul3)
    rpsi = basis3(i, j, k,  x - ddd, y, z,  mul1, mul2, mul3)
    dxpsi = (fpsi + rpsi - 2*psi)/(ddd)**2*h12/2
    
    fpsi = basis3(i, j, k, x, y+ddd, z,  mul1, mul2, mul3)
    rpsi = basis3(i, j, k, x, y-ddd, z,  mul1, mul2, mul3)
    dypsi = (fpsi + rpsi - 2*psi)/(ddd)**2*h12/2
    
    fpsi = basis3(i, j, k, x, y, z+ddd,  mul1, mul2, mul3)
    rpsi = basis3(i, j, k, x, y, z-ddd,  mul1, mul2, mul3)
    dzpsi = (fpsi + rpsi - 2*psi)/(ddd)**2*h12/2
    
    
    #print(psi, fpsi, rpsi)
    return psi, dxpsi, dypsi, dzpsi

