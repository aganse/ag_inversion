import math
import numpy as np
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator


def convert_deriv_to_mag(J,m):
    """
    Convert jacobian matrix of complex derivatives of predicted values wrt model params
    to a jacobian matrix of derivatives of the magnitudes of the predicted values wrt
    model parameters.
    Requires inputs of both jacobian matrix J and model params m at which J is valid.
    (ie since typically this is used for a locally-linearized nonlinear model.)
    Assumes dims NxM for N predicted values and M model parameters
    """
    # Hm, this expression is quite off when comparing to findiffs, rederive:
    # Jmag = [np.conj(m) * J  +  m * np.conj(J)]  /2/np.abs(m)
    Jmag = np.add( np.dot(np.conj(m.T),J) , np.dot(m.T,np.conj(J)) ) /2/np.abs(m)
    Jmag = np.real(Jmag)  # just stripping off the +0.j's and setting type to real
    # That /2/np.abs(m) is a col vector of same length as m, and is supposed to
    # divide element-wise into every column the matrix via Python's broadcasting.
    return Jmag


def separate_cplx(x):
    # (check if x is complex here...)
    if x.ndim==1:
        x2 = np.concatenate((np.real(x),np.imag(x)),axis=0)
    elif x.ndim==2:
        x2 = np.concatenate((np.real(x),np.imag(x)),axis=0)
    else:
        print('separate_cplx: error: input dim is not 1 or 2.')
    return x2


def cplxMSE(a,b):
    # mean squared error that can handle complex data
    r = np.subtract(a,b)
    mse = np.dot( np.conjugate(r).T, r )
    mse = np.real(mse).item()  # result above was real but had +0j in a complex
                               # typed var so take real part, and then came back
                               # as singleton array so item() converts that to
                               # scalar type.
    return mse


def jacfindiff(fwdfunc,x,dx=1.0e-8):
    # first order fwddiffs calc for jacobian matrix estimation
    M = len(x)
    dx = dx * np.ones((M,1))
    x = x.reshape(len(x),1)
    # Now make dx and exactly representable number vis a vis machine precision
    # so that all error in derivs are from numerator (see Num Rec 2nd ed, sec 5.7)
    temp = x+dx;  # note temp is vector of same length as model vector
    dx = temp-x;  # Believe it or not this is actually a slightly different dx now,
                  # different by an error of order of machine precision.
                  # This effect may or may not make it into fwdprob input which
                  # could have limited input precision, but in any case will have
                  # an effect via denominator at end of this script.
    mask=np.eye(M);
    F = fwdfunc(x)
    J = np.ndarray(shape=(len(F),M), dtype=float)
    for j in range(M):
        d = np.multiply(dx, mask[:,j].reshape(M,1))
        xtmp = x + d
        J[:,j] = (fwdfunc(xtmp) - F) / dx[j]
        print('.', end='')
    return J


#def check_derivs()



"""
Inversion model classes based on LinearRegression from scikit-learn.
It's almost not worth importing skl, but it does have multiple solution methods.
For inversion, the X is no longer from the data, it's the Jacobian matrix of
derivatives, and the model coeffs are the desired inverted sal profiles.
Replacing predict() method to use the X (derviatives) at the model coeffs
solution, with those coeffs, to produce the predicted data.
"""

class InvT0(BaseEstimator):
    """ Linear inversion using Tikhonov regularization.
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, fwd_deriv_code, minit, alpha=0.01, max_iter=5, tol=0.001,
                 usefindiff=False, showplot=True, verbose=True):
        self.fwd_deriv_code = fwd_deriv_code
        self.minit = minit
        self.max_iter = max_iter  # only relevant after expanding this to nonlinear
        self.alpha = alpha
        self.tol = tol  # only relevant after expanding this to nonlinear
        self.usefindiff = usefindiff
        self.showplot = showplot
        self.verbose = verbose
        #super().__init__(fit_intercept=False, copy_X=True)
        #super(InvT0, self).__init__(self, fit_intercept=False, copy_X=True)

    def fit(self, ymeas, mtrue=None, alg='optls'):   # alg: {'optls','mine'}
        """A reference implementation of a fitting function
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """

        # [ X ]*dm = [ dy ]
        # [ a ]      [  0 ]   <-- using built-in Ridge model does this
        #
        # [ X ]*dm = [ dy  ]
        # [-a ]      [ a*m ] <-- but I want this for iterated nonlin problem
        #
        # [ X  ]*dm = [   dy  ]
        # [-aL ]      [ a*L*m ] <-- and more generally I want this (higher-order Tihk)
        #
        # which can be rewritten:
        #   G * dm  = D         (and then loop that from m0 with m=m+dm...)

        # X is the Jacobian matrix of derivs of predicted data points wrt model
        # params m, as given by ypred,X=self.fwd_deriv_code(m)...

        if alg=='optls':
            # (in progress...)
            # x,cov_x,infodict,mesg,ier =
            # leastsq(func, x0, args=(), Dfun=None, full_output=0, col_deriv=0,
            # ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, maxfev=0, epsfcn=None,
            # factor=100, diag=None)
            #
            # m,cost,misfit,modelnorm,norm(dm),testMSE

        elif alg=='mine':
            cost = []
            m = self.minit
            mlen = len(m)
            if self.verbose:
                print(' alpha      cost       norm(dd)    norm(dm)   tol')
            for i in range(self.max_iter):
                ypred,X = self.fwd_deriv_code(m)  # m: model params vector, X: derivs matrix
                if self.usefindiff:
                    def tmpfwdcode(m):
                        return np.squeeze(self.fwd_deriv_code(m)[0])
                    X = jacfindiff(tmpfwdcode,m,dx=1.0e-6)  # dx=1.0e-6 is problem dependent!
                G = np.concatenate((X, -self.alpha*np.eye(mlen)),axis=0)
                D = np.concatenate((ymeas-ypred, self.alpha*m),axis=0)
                misfit = cplxMSE(ymeas, ypred)
                modelnorm = norm(m)**2
                cost.append(misfit + pow(self.alpha,2)*modelnorm)
                dm,res,rnk,sv = lstsq(G,D)
                m = m + dm
                if self.verbose:
                    print('%6.1g  %10.2g  %10.2g  %10.2g  %6.3f' %
                        (self.alpha, cost[-1], norm(ymeas-ypred), norm(dm), self.tol))
                if norm(dm) < self.tol:
                   break
                self.G = G
                self.ypred = ypred
                if mtrue is not None:
                    testMSE = cplxMSE(m,mtrue)
                else:
                    testMSE = npl.nan
            if self.showplot:
                f, ax = plt.subplots(1, 2, figsize=(11,4))
                # plot the cost (ie loss) per iterations:
                ax[0].semilogy(cost,'.-')  # (last element of cost)
                ax[0].grid()
                ax[0].set_ylabel('cost')
                ax[0].set_xlabel('iterations')
                ax[0].set_title('Loss history  (misfit + alpha*modelnorm)')
                # plot the init, true, final, and evolution of model params:
                #print('m',np.squeeze(m.T))
                ax[1].plot(1/mtrue,'k')
                ax[1].plot(1/self.minit,'g')
                ax[1].plot(1/m,'r')
                ax[1].grid()
                ax[1].set_ylabel('model value')
                ax[1].set_xlabel('indep var')
                ax[1].set_title('Model vectors')

        return m,cost,misfit,modelnorm,norm(dm),testMSE


    def get_hyperparams(self):
        return (self.max_iter, self.tol, self.alpha)
