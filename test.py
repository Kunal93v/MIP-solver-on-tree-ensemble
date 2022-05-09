
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from prepare import *
from MIP_solver import *
#get input rf from sample "concrete"
df = pd.read_csv('concrete.csv')
df=df.drop(df.columns[0], axis=1)
X_variable=df.iloc[:,:-1]
Y_variable=df.iloc[:,-1]
labels=np.array(Y_variable)
feature_list = list(X_variable.columns)
features = np.array(X_variable)
rf = RandomForestRegressor(n_estimators=2)
rf.fit(features,labels)
flag=1 #regression tree
#equal weighted
weight=np.zeros(rf.n_estimators)
for i in range(rf.n_estimators):
    weight[i]=1/rf.n_estimators
trees=list()
trees=get_input(rf)
lama=np.zeros(len(trees))
for i in range(len(trees)):
    lama[i]=weight[i]    
#[alpha,beta,gamma]=initial_solver(trees,flag)  
alpha={}
beta={}
gamma={}
for i in range(len(trees)):
    for j in splits(trees,i):
        #create variables
        alpha[i,j]=0
        beta[i,j]=0
for i in range(len(trees)):
    gamma[i]=0
    for j in leaves(trees,i):
        gamma[i] = max(gamma[i],prediction(trees,i,j,flag))
def add_constraint(model, where):
    if where == GRB.Callback.MIPSOL:
        #sol_X = model.cbGetSolution([model._vars_X[i] for i in range(len(model._vars_X))])
        sol_theta=model.cbGetSolution([model._vars_theta[i] for i in range(len(model._vars_theta))])
        sol_X_one={}
        for i in total_split_variable(trees):
            for j in range(K(trees,i)):                
                sol_X_one[i,j]=model.cbGetSolution(model._vars_X_one[i,j])
        alpha_new={}
        beta_new={}
        gamma_new={}
        for i in range(len(trees)):
            l_optimal=GETLEAF(trees,i,sol_X_one)
            for j in splits(trees,i):
                temp1=0
                if j in as_right_leaf(trees,i,l_optimal):
                    for l in left_leaf(trees,i,j):
                        temp1=max(temp1,(prediction(trees,i,l,flag)-prediction(trees,i,l_optimal,flag)))
                alpha_new[i,j]=temp1
                temp2=0
                if j in as_left_leaf(trees,i,l_optimal):
                    for l in right_leaf(trees,i,j):
                        temp2=max(temp2,prediction(trees,i,l,flag)-prediction(trees,i,l_optimal,flag))
                beta_new[i,j]=temp2
            gamma_new[i]=prediction(trees,i,l_optimal,flag)
        for i in reversed(range(len(trees))):
            #expr=quicksum(alpha_new[i,s]*x(trees,sol_X,i,s) for s in splits(trees,i)) + quicksum(beta_new[i,s]*(1-x(trees,sol_X,i,s)) for s in splits(trees,i)) + gamma_new[i] - sol_theta[i]
            expr=0
            for s in splits(trees, i):
                expr=expr+alpha_new[i,s]*sol_X_one[V(trees,i,s),C(trees,i,s)]+beta_new[i,s]*(1-sol_X_one[V(trees,i,s),C(trees,i,s)])
            expr=expr+gamma_new[i]-sol_theta[i]            
            if expr < -1e-5:                
                model.cbLazy(quicksum(alpha_new[i,s]*model._vars_X_one[V(trees,i,s),C(trees,i,s)] for s in splits(trees,i)) + quicksum(beta_new[i,s]*(1-model._vars_X_one[V(trees,i,s),C(trees,i,s)]) for s in splits(trees,i)) + gamma_new[i] - model._vars_theta[i] >= 0) 
                print("Find a violated constraint!")
                break 
                #model.cbLazy(quicksum(alpha_new[i,s]*x(trees,model._vars_X,i,s) for s in splits(trees,i)) + quicksum(beta_new[i,s]*(1-x(trees,model._vars_X,i,s)) for s in splits(trees,i)) + gamma_new[i] - model._vars_theta[i] >= 0)


#create a new model
m = Model("tree_ensemble")

#create variables
X={}
theta={}
X_one={}

for i in total_split_variable(trees):
    X[i]=m.addVar(lb=-GRB.INFINITY, name='X'+str(i))
    for j in range(K(trees,i)):
        X_one[i,j]=m.addVar(vtype=GRB.BINARY, name='X_one'+str(i)+'_'+str(j))
for i in range(len(trees)):
    theta[i]=m.addVar(lb=-GRB.INFINITY, name='theta' + str(i))
m.update()

# Set objective
m.setObjective(quicksum(lama[i]*theta[i] for i in range(len(trees))), GRB.MAXIMIZE)
m.update()

# Add constraint
for i in range(len(trees)):
    m.addConstr(quicksum(alpha[i,s]*X_one[V(trees,i,s),C(trees,i,s)] for s in splits(trees,i)) + quicksum(beta[i,s]*(1-X_one[V(trees,i,s),C(trees,i,s)]) for s in splits(trees,i)) + gamma[i] - theta[i] >= 0) 

for i in range(len(trees)):
    for j in splits(trees,i):
        m.addConstr((X_one[V(trees,i,j),C(trees,i,j)] == 1) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] <= 0) )
        m.addConstr((X_one[V(trees,i,j),C(trees,i,j)] == 0) >> (X[V(trees,i,j)] - split_values(trees,V(trees,i,j))[C(trees,i,j)] >= 1e-5) )

for i in total_split_variable(trees):
    for j in range(K(trees,i)-1):
        m.addConstr(X_one[i,j] - X_one[i,j+1] <= 0)

m.update()

#m._vars_X=X
m._vars_X_one=X_one
m._vars_theta=theta
m.params.LazyConstraints = 1
m.optimize(add_constraint)


optimal_value=np.zeros(len(X))
for i in range(len(X)):
    optimal_value[i]=X[i].x
print(optimal_value)
