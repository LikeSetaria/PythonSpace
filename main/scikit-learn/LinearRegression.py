from sklearn import linear_model
from sklearn.linear_model import Ridge

clf=linear_model.LinearRegression()
clf.fit([[0,0],[0,0],[1,1]],[0,.1,1])
Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
normalize=False, random_state=None, solver='auto', tol=0.001)
clf.coef_array([ 0.34545455, 0.34545455])