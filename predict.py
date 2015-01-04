##
import numpy as np
import patsy
import scipy
import statsmodels.api as sm

##
def predict(L, formula, data, level=0.95, interval="prediction", model_matrix = False):
    """
    L is either a model matrix or a data frame 
    of the same structure like the data argument. 
    formula and data describe the model.          
    interval: "prediction" of "confidence"
    """
    
    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    model = sm.OLS(y, X).fit()

    if not model_matrix:
        L = patsy.dmatrices(formula, L, return_type="matrix")[1] # same columns like the model matrix now
    xtx_pinv = np.linalg.pinv(X.T.dot(X))
    
    if interval=="confidence":
        se = np.array([np.sqrt(model.mse_resid*vect.dot(xtx_pinv).dot(vect.T)) for vect in L])
    else: 
        se = np.array([np.sqrt(model.mse_resid*(1+vect.dot(xtx_pinv).dot(vect.T))) for vect in L])
    
    t = scipy.stats.t.ppf((level+1)/2, model.df_resid)
    point_estimates = np.array([(vect*model.params).sum() for vect in L])
    lower = point_estimates - t*se
    upper = lower + 2*t*se
    return np.hstack([lower.reshape(-1,1), upper.reshape(-1,1)])

##
plt.figure()
plt.plot(predictions[:,0], 'r--')
plt.plot(predictions[:,1], 'r--')
plt.plot(confidence[:,0], 'b-')
plt.plot(confidence[:,1], 'b-')
plt.plot((confidence[:,0]+confidence[:,1])/2, 'ko')
plt.plot((confidence[:,0]+confidence[:,1])/2, 'k-')
plt.show()
##



