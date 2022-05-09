
import pandas as pd
import numpy as np
from prepare import *
from gurobipy import *
from MIP_solver import *
from MIP_solver_2 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#y as a function of x
def fun_high(x):
    return -(x[:,0]-5)**2-(x[:,1]-5)**2-(x[:,2]-5)**2-(x[:,3]-5)**2-(x[:,4]-5)**2+125      
  
  
seed=1
np.random.seed(seed)
x=5*np.random.rand(300,5)
x_add=2*np.random.rand(300,5)+1
x=np.vstack((x, x_add))
y=fun_high(x)


flag=1 #regression tree
rf=get_rf(x,y,3,flag)


trees=list()
trees=get_input(rf)


eta_1=0
eta_2=125
bisection(trees,flag,eta_1,eta_2,80)
