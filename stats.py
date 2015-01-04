import statsmodels.api as sm
import patsy
import scipy
import numpy as np

def lm(formula, data):
	"""
	this function takes a patsy formula and
	a pandas dataframe. the names of variables
	in the formula are columns of the dataframe
	"""
	y, X = patsy.dmatrices(formula, data, return_type='dataframe')
	results = sm.OLS(y, X).fit()
	print(results.summary())
	return results


def data(dataname = None, package = None, cache = False):
	"""
	loads R dataset called 'dataname' from package called 'package'
	"""
	#if dataname == None and data == None:
	#  from rpy2.robjects import r
	#  print(r.data())
	return sm.datasets.get_rdataset(dataname = dataname, package = package, cache = cache).data


def submodel(model_formula, submodel_formula, data):
	"""
	specify model and submodel formulas and model data.
	Function tests submodel using F test.
	Returns the value of F and the pvalue of the test.
	"""
	y1, X1 = patsy.dmatrices(model_formula, data, return_type='dataframe')
	y2, X2 = patsy.dmatrices(submodel_formula, data, return_type='dataframe')
	model = sm.OLS(y1, X1).fit()
	submodel = sm.OLS(y2, X2).fit()
	F=((submodel.ssr-model.ssr)/(submodel.df_resid-model.df_resid))/model.mse_resid
	df1, df2 = submodel.df_resid-model.df_resid, model.df_resid
	pvalue = 1-scipy.stats.f.cdf(F, df1, df2)
	message = """
	Null hypothesis: submodel holds
	F statistic: %(F)s
	df1, df1 = %(df1)s, %(df2)s
	p-value: %(pvalue)s
	""" % {'F': F, 'df1': int(df1), 'df2': int(df2), 'pvalue': pvalue}
	print(message)
	return F, pvalue


def chisq_test(observed):
	"""
	performs a chi squared test of independence
	on a contingency table (NumPy array). Returns
	the test statistic and the p-value of the test. 
	"""
	n, k = observed.shape
	row = observed.sum(axis=0).reshape(1,-1)
	col = observed.sum(axis=1).reshape(-1,1)
	expected = np.dot(col, row)/observed.sum()
	#chi2, pvalue = scipy.stats.mstats.chisquare(observed.ravel(), expected.ravel(), ddof = n+k-2)
	chi2 = (((observed-expected)**2)/expected).sum()
	pvalue = 1-scipy.stats.chi2.cdf(chi2, (n-1)*(k-1))
	message = """
	Performing the test of independence in	a contingency table.
	test statistic: %(chi2)s
	degrees of freedom: %(df)s
	p-value: %(pvalue)s
	""" % {'chi2': chi2, 'df': (n-1)*(k-1), 'pvalue': pvalue}
	print(message)
	warning = """
	Warning message:
	Chi-squared approximation may be incorrect
	"""
	if expected.min() < 5:
		print(warning)
	return chi2, pvalue


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
