import statsmodels.api as sm
import patsy

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
  data called dataname from package called package
  """
  #if dataname == None and data == None:
  #  from rpy2.robjects import r
  #  print(r.data())
  return sm.datasets.get_rdataset(dataname = dataname, package = package, cache = cache).data
