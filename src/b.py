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
    lambdas = np.logspace(-3, 2, 6)

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
        ax[1].set_title('$\\mathrm{Ridge}$',fontsize=14)
        ax[1].tick_params(labelsize=16)
    
        im=ax[2].imshow(J_lasso,**cmap_args)
        ax[2].set_title('$\\mathrm{LASSO}$',fontsize=14)
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
    plt.semilogx(lambdas, train_mse_ridge,'--r',label='Train (Ridge)')
    plt.semilogx(lambdas, test_mse_ridge,':r',label='Test (Ridge)')
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
