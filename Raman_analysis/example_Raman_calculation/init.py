import flow
import numpy as np
import signac


project = signac.init_project(name='project', root="./")

adds = []
amoeba_multiply = 1.07 #Factor for using Amoeba instead of SPC/E



for end in [-2, 2, 6]:
    for i in range(0, 2000, 5):
        for kstretch in [0.39*amoeba_multiply]:
            for kb1 in np.array([-1e-3*amoeba_multiply]):
                for kb2 in np.array([2.8e-5*amoeba_multiply]):
                    for kbendstd in np.array([3.7e-1]):
                        sp  = dict(i=i, kstretch=kstretch, kb1=kb1, kb2=kb2, kbendstd=kbendstd, nrandos=2000, end=end, leng=4.0)
                        job = project.open_job(sp)
                        print(job.id)
                        job.init()
                        job.document['Finished'] = 'no'
