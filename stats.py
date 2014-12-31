import statsmodels.api as sm
import patsy
import scipy

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
