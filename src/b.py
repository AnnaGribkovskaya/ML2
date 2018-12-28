import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse as sp
import pickle
import seaborn 
import warnings
from a import ising_energies, generate_data_set

# system size
L=40
n_samples=600
# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))

def compare_regressions(states,L,n_samples):
    # Set data
    data, X_train, X_test, Y_train, Y_test = generate_data_set(states,L,n_samples)

    #OLS
    ols=linear_model.LinearRegression()
    coefs_ols = []
    train_score_OLS = []
    test_score_OLS = []
    train_mse_OLS = []
    test_mse_OLS = []

    #Ridge
    ridge=linear_model.Ridge()
    coefs_ridge = []
    train_score_ridge = []
    test_score_ridge = []
    train_mse_ridge = []
    test_mse_ridge = []

    #LASSO
    lasso = linear_model.Lasso()
    train_score_lasso = []
    test_score_lasso = []
    train_mse_lasso = []
    test_mse_lasso = []
    coefs_lasso=[]

    #Set LAMBDAS
    lambdas = np.logspace(-4, 5, 10)

    seaborn.set(style="white", context="notebook", font_scale=1.5, rc={"axes.grid": True, "legend.frameon": False, "lines.markeredgewidth": 1.4, "lines.markersize": 10})
    seaborn.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})
    plt.style.use('fivethirtyeight')
    seaborn.set_style({'grid.color': '.5'})
    seaborn.set_style('whitegrid')


    for alpha in lambdas: 

        # OLS
        ols.fit(X_train, Y_train) 
        coefs_ols.append(ols.coef_) 
        train_score_OLS.append(ols.score(X_train, Y_train))
        test_score_OLS.append(ols.score(X_test,Y_test))
        train_mse_OLS.append(mean_squared_error(Y_train, ols.predict(X_train)))
        test_mse_OLS.append(mean_squared_error(Y_test, ols.predict(X_test)))
        
        #  Ridge 
        ridge.set_params(alpha=alpha) 
        ridge.fit(X_train, Y_train) 
        coefs_ridge.append(ridge.coef_) 
        train_score_ridge.append(ridge.score(X_train, Y_train))
        test_score_ridge.append(ridge.score(X_test,Y_test))
        train_mse_ridge.append(mean_squared_error(Y_train, ridge.predict(X_train)))
        test_mse_ridge.append(mean_squared_error(Y_test, ridge.predict(X_test)))
        
        #  LASSO 
        lasso.set_params(alpha=alpha) 
        lasso.fit(X_train, Y_train) 
        coefs_lasso.append(lasso.coef_) 
        train_score_lasso.append(lasso.score(X_train, Y_train))
        test_score_lasso.append(lasso.score(X_test,Y_test))
        train_mse_lasso.append(mean_squared_error(Y_train, lasso.predict(X_train)))
        test_mse_lasso.append(mean_squared_error(Y_test, lasso.predict(X_test)))


        ''' Plotting interaction pmatrix J_ij '''
        J_ols=np.array(ols.coef_).reshape((L,L))
        J_ridge=np.array(ridge.coef_).reshape((L,L))
        J_lasso=np.array(lasso.coef_).reshape((L,L))
        cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')

        fig, ax = plt.subplots(nrows=1, ncols=3)
    
        ax[0].imshow(J_ols,**cmap_args)
        ax[0].set_title('$\\mathrm{OLS}$',fontsize=14)
        ax[0].tick_params(labelsize=16)
    
        ax[1].imshow(J_ridge,**cmap_args)
        ax[1].set_title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(alpha),fontsize=14)
        ax[1].tick_params(labelsize=16)
    
        im=ax[2].imshow(J_lasso,**cmap_args)
        ax[2].set_title('$\\mathrm{LASSO},\ \\lambda=%.4f$' %(alpha),fontsize=14)
        ax[2].tick_params(labelsize=16)
    
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=fig.colorbar(im, cax=cax)
    
        cbar.ax.set_yticklabels(np.arange(-1.0, 1.0+0.25, 0.25),fontsize=14)
        cbar.set_label('$J_{i,j}$',labelpad=-40, y=1.12,fontsize=16,rotation=0)
    
        fig.subplots_adjust(right=2.0)
        plt.draw()
        plt.show()
        fig.savefig("lambda=" + str(alpha) + ".pdf", bbox_inches='tight')

    '''
    Make plots for TEST and TRAIN R2 results for different lambdas 
    '''
    plt.semilogx(lambdas, train_score_OLS, '--k',label='Train (OLS)')
    plt.semilogx(lambdas, test_score_OLS,':k',label='Test (OLS)')
    plt.semilogx(lambdas, train_score_ridge,'--r',label='Train (Ridge)')
    plt.semilogx(lambdas, test_score_ridge,':r',label='Test (Ridge)')
    plt.semilogx(lambdas, train_score_lasso, '--g',label='Train (LASSO)')
    plt.semilogx(lambdas, test_score_lasso, ':g',label='Test (LASSO)')
    fig = plt.gcf()
    fig.set_size_inches(12.0, 6.0)
    plt.legend(loc='lower left',fontsize=12)
    plt.ylim([-0.01, 1.01])
    plt.xlim([min(lambdas), max(lambdas)])
    plt.xlabel(r'$\lambda$',fontsize=12)
    plt.ylabel('R2',fontsize=12)
    plt.tick_params(labelsize=12)
    plt.draw()
    plt.show()
    fig.savefig("1.pdf", bbox_inches='tight')

    '''
    Make plots for TEST and TRAIN MSE results for different lambdas 

    '''
    plt.semilogx(lambdas, train_mse_OLS, '--k',label='Train (OLS)')
    plt.semilogx(lambdas, test_mse_OLS,':k',label='Test (OLS)')
    plt.semilogx(lambdas, train_mse_ridge,'--r',label='Train (Ridge)')#,linewidth=1)
    plt.semilogx(lambdas, test_mse_ridge,':r',label='Test (Ridge)')#,linewidth=1)
    plt.semilogx(lambdas, train_mse_lasso, '--g',label='Train (LASSO)')
    plt.semilogx(lambdas, test_mse_lasso, ':g',label='Test (LASSO)')
    fig = plt.gcf()
    fig.set_size_inches(10.0, 6.0)
    plt.legend(loc='upper left',fontsize=12)
    plt.xlim([min(lambdas), max(lambdas)])
    plt.xlabel(r'$\lambda$',fontsize=12)
    plt.ylabel('MSE',fontsize=12)
    plt.tick_params(labelsize=12)
    plt.draw()
    plt.show()
    fig.savefig("2.pdf", bbox_inches='tight')



compare_regressions(states,L,n_samples)


'''


from collections import OrderedDict
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
#%matplotlib inline

def bootstrap(xData, yData,  R, lmbda):
    n = xData.shape[0] 
    inds = np.arange(n); 

    mseMatrixUnknownF =  OrderedDict()
    sdMatrixUnknownF =  OrderedDict()
    bias2MatrixUnknownF =  OrderedDict()
    totalMatrixUnknownF =  OrderedDict()
    residualDictUnknownF = OrderedDict()
    
    models = ['ridge', 'lasso']
    
    for i in models:
        mseMatrixUnknownF[i] = np.zeros(n)
        sdMatrixUnknownF[i] = np.zeros(n)
        bias2MatrixUnknownF[i] = np.zeros(n)
        totalMatrixUnknownF[i] = np.zeros(n)
        residualDictUnknownF[i] = OrderedDict()
        for j in range(n):
            residualDictUnknownF[i][j] = []    
                                                                                                                                        
    for i in range(R):
        idxTrain = np.random.randint(0,n,n)
        idxValid = np.setdiff1d(np.union1d(inds, idxTrain ), np.intersect1d(inds, idxTrain))
        
        x_train = xData[idxTrain]
        y_train = yData[idxTrain]
        x_valid = xData[idxValid]
        y_valid = yData[idxValid]
        
        ridge.set_params(alpha=lmbda) # set regularisation parameter
        ridge.fit(x_train, y_train) # fit model
        yPredictRidge = ridge.predict(x_valid)
        
         
        lasso.set_params(alpha=lmbda) # set regularisation parameter
        lasso.fit(x_train, y_train) # fit model
        yPredictLasso = lasso.predict(x_valid)
        
        for i in range(len(idxValid)):
            residualDictUnknownF[models[0]][i].append(y_valid[i] -yPredictRidge[i])
            residualDictUnknownF[models[1]][i].append(y_valid[i] -yPredictLasso[i])
            
    mseUnknownF = OrderedDict()
    sdUnknownF = OrderedDict()
    bias2UnknownF = OrderedDict()
    totalUnknownF = OrderedDict()
        
    for model in models:
        for i in range(n):
            mseMatrixUnknownF[model][i] = np.mean([(residualDictUnknownF[model][i][j])**2 for j in range(len(residualDictUnknownF[model][i]))])#np.mean( (residualDictUnknownF[key])**2 )
            sdMatrixUnknownF[model][i] = np.var( residualDictUnknownF[model][i] )
            bias2MatrixUnknownF[model][i] = ( np.mean(residualDictUnknownF[model][i]) )**2
            totalMatrixUnknownF[model][i] = sdMatrixUnknownF[model][i] + \
    bias2MatrixUnknownF[model][i]
        mseUnknownF[model] = np.nanmean(mseMatrixUnknownF[model])
        sdUnknownF[model] = np.nanmean(sdMatrixUnknownF[model])
        bias2UnknownF[model] = np.nanmean(bias2MatrixUnknownF[model])
        totalUnknownF[model] = np.nanmean(totalMatrixUnknownF[model])
            
    return mseUnknownF, sdUnknownF, bias2UnknownF, totalUnknownF

ols=linear_model.LinearRegression()
ridge=linear_model.Ridge()
lasso = linear_model.Lasso()
lmbdas = np.logspace(-4, 5, 10)
R = 25
xData = X_train
yData = Y_train
mses, sds, biases, totals = [], [], [], []
for lmbda in lmbdas:
    mseUnknownF, sdUnknownF, bias2UnknownF, totalUnknownF = bootstrap(xData, yData,  R, lmbda)   
    mses.append(mseUnknownF)
    sds.append(sdUnknownF)
    biases.append(bias2UnknownF)
    totals.append(totalUnknownF)

msePlot = OrderedDict()
varPlot = OrderedDict()
biasPlot = OrderedDict()
totalPlot = OrderedDict()

models = ['ridge', 'lasso']
for model in models:
    msePlot[model] = []
    varPlot[model] = []
    biasPlot[model] = []
    totalPlot[model] = []


for model in models:
    for i in range(len(mses)):
        msePlot[model].append(mses[i][model])
        varPlot[model].append(sds[i][model])
        biasPlot[model].append(biases[i][model])
        totalPlot[model].append(totals[i][model])

#print(mseUnknownF, sdUnknownF, bias2UnknownF, totalUnknownF)
fig, ax = plt.subplots()
ax.semilogx(lmbdas, varPlot[models[0]], 'b',label='Var Ridge')
ax.semilogx(lmbdas, biasPlot[models[0]], '--b',label='Bias^2 Ridge')
ax.semilogx(lmbdas, varPlot[models[1]], 'g',label='Var Lasso')
ax.semilogx(lmbdas, biasPlot[models[1]], '--g',label='Bias^2 Lasso')
ax.set_xlabel(r'$\lambda$')
ax.set_title('Bias variance \n 1D regressions')
#fig.legend(loc='right',fontsize=16)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)\
, fontsize = 16)
plt.draw()
#plt.show()
fig.savefig("3.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.semilogx(lmbdas, np.array(varPlot[models[0]])/np.array(msePlot[models[0]]), 'b',label='Var Ridge')
ax.semilogx(lmbdas, np.array(biasPlot[models[0]])/np.array(msePlot[models[0]]), '--b',label='Bias^2 Ridge')
ax.semilogx(lmbdas, np.array(varPlot[models[1]])/np.array(msePlot[models[1]]), 'g',label='Var Lasso')
ax.semilogx(lmbdas, np.array(biasPlot[models[1]])/np.array(msePlot[models[1]]), '--g',label='Bias^2 Lasso')
ax.set_xlabel(r'$\lambda$')
ax.set_title('Bias variance shares \n 1D regressions')
#fig.legend(loc='right',fontsize=16)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)\
, fontsize = 16)
plt.draw()
#plt.show()
fig.savefig("4.pdf", bbox_inches='tight')
'''
