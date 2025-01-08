import flow
import numpy as np
import signac
import os

class MyProject(flow.FlowProject):
    pass

@MyProject.label
def finished(job):
    with job:
        job.isfile("dump.lammpstrj")


@MyProject.post.isfile('full_run.py')
@MyProject.operation
def initialize(job):
    with job:
       os.system("cp ../../pressjob.sh .")
       os.system("cp ../../necessary_functions.py .")
       os.system("cp ../../full_run.py .")
       os.system("cp ../../polarizability_krr.joblib .")
       os.system("cp ../../one_body_ridge.joblib .")
       os.system("cp ../../one_body_shifts.npy .")

       job.document['Finished'] = 'no'
       
       submit_file  = """

from full_run import compute_distribution

compute_distribution({i}, {kstretch}, {kb1}, {kb2}, {kbendstd}, {nrandos}, {end}, {leng})

""".format(
       i=job.sp.i,
       kstretch=job.sp.kstretch,
       kb1=job.sp.kb1,
       kb2=job.sp.kb2,
       kbendstd=job.sp.kbendstd,
       nrandos=job.sp.nrandos,
       end=job.sp.end,
       leng=job.sp.leng)
       f = open('submit.py', 'w')
       f.write(submit_file)
       f.close()
@MyProject.pre.isfile('full_run.py')
@MyProject.post.isfile('activities.npy')
@MyProject.operation
def submit(job):
    with job:
        os.system("sbatch pressjob.sh")
        job.document['Finished'] = 'yes'


project = MyProject.get_project(root="./")

if __name__ == '__main__':
    MyProject().main()
