import numpy as np
import patsy
import scipy

def predict(L, level, model, formula, data, model_matrix = False):
    """
    L is a model matrix with columns names like in model. 
    model is a fitted model returned by lm.
    formula and data were fed to lm.          
    """
    if model_matrix:
        pass
    else:
        # prediction interval for new observations:
	    y, X = patsy.dmatrices(formula, data)
	    X = np.asarray(X)
	    xtx_pinv = np.linalg.pinv(X.T.dot(X))
	    se = np.array([np.sqrt(model.mse_resid*(1+vect.dot(xtx_pinv).dot(vect.T))) for vect in L])
	    t = scipy.stats.t.ppf((level+1)/2, model.df_resid)
	    lower = model.predict(X) - t*se
	    upper = lower + 2*t*se
    return np.hstack([lower.reshape(-1,1), upper.reshape(-1,1)])
        

