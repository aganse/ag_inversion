import math
import numpy as np
from numpy.linalg import norm, lstsq, inv, cholesky
from scipy.linalg import toeplitz
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator


def convert_deriv_to_mag(J,dpred):
    """
    Convert jacobian matrix of complex derivatives of predicted values wrt model params
    to a jacobian matrix of derivatives of the magnitudes of the predicted values wrt
    model parameters.
    Requires inputs of both jacobian matrix J and predicted data dpred associated with J.
    (ie since typically this is used for a locally-linearized nonlinear model.)
    Assumes dims NxM for N predicted values and M model parameters
    """
    # FIXME: reverify this equation!
    # Jmag = [np.conj(m) * J  +  m * np.conj(J)]  /2/np.abs(m)
    Jmag = np.add( np.dot(np.conj(dpred.T),J) , np.dot(dpred.T,np.conj(J)) ) /2/np.abs(dpred)
    Jmag = np.real(Jmag)  # just stripping off the +0.j's and setting type to real
    # That /2/np.abs(dpred) divides element-wise into every column the matrix via Python's broadcasting.
    return Jmag


def convert_deriv_to_inv(J,dpred):
    """
    Convert jacobian matrix of complex derivatives of predicted values wrt model params
    to a jacobian matrix of derivatives of the inverse of the predicted values wrt
    model parameters.
    Requires inputs of both jacobian matrix J and predicted data dpred associated with J.
    (ie since typically this is used for a locally-linearized nonlinear model.)
    Assumes dims NxM for N predicted values and M model parameters
    """
    Jinv = -J /dpred**2
    # That /dpred**2 divides element-wise into every column the matrix via Python's broadcasting.
    return Jinv


def convert_deriv_to_log10(J,dpred):
    """
    Convert jacobian matrix of complex derivatives of predicted values wrt model params
    to a jacobian matrix of derivatives of the log10 of the predicted values wrt
    model parameters.
    Requires inputs of both jacobian matrix J and predicted data dpred associated with J.
    (ie since typically this is used for a locally-linearized nonlinear model.)
    Assumes dims NxM for N predicted values and M model parameters.
    Assumes dpred and J are already real-valued.
    """
    Jlog = np.log10(np.exp(1.0)) * J /dpred
    # That /dpred divides element-wise into every column the matrix via Python's broadcasting.
    return Jlog


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
    mse = mse/r.size
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


def fd_mtx(mlen,type='fwd',order=2,bounds=False):
    """Produces finite difference matrices of specified type & order.
    type = {'fwd','bkwd','ctr'}
    order = {1,2} for first or second diffs
    bounds = {True,False} include/not the boundary rows at top or bottom.
             If True then matrix is square.
    """
    r = np.zeros(mlen)
    c = np.zeros(mlen)
    if type=='fwd':
        if order==1:
            r[0] = -1
            r[1] = 1
            c[0] = -1
        elif order==2:
            r[0] = 1
            r[1] = -2
            r[2] = 1
            c[0] = 1
    elif type=='bkwd':
        if order==1:
            r[0] = 1
            c[0] = 1
            c[1] = -1
        elif order==2:
            r[0] = 1
            c[0] = 1
            c[1] = -2
            c[2] = 1
    elif type=='ctr':
        if order==1:
            r[1] = .5
            c[1] = -.5
        elif order==2:
            r[0] = -2
            r[1] = 1
            c[0] = -2
            c[1] = 1
    T = toeplitz(c,r)
    if bounds==False:
        if order==1 and type=='fwd':
            T = np.delete(T, (mlen-1), axis=0)
        elif order==1 and type=='bkwd':
            T = np.delete(T, (0), axis=0)
        elif order==1 and type=='ctr':
            T = np.delete(T, (0,mlen-1), axis=0)
        elif order==2 and type=='fwd':
            T = np.delete(T, (mlen-2,mlen-1), axis=0)
        elif order==2 and type=='bkwd':
            T = np.delete(T, (0,1), axis=0)
        elif order==2 and type=='ctr':
            T = np.delete(T, (0,mlen-1), axis=0)
        # Need the rest of those here but those will get things started...
    return T


def create_findiff_mtx(mlen,beta):
    L = fd_mtx(mlen,'fwd',2,bounds=False)  # 2nd finite diff matrix
    L = np.delete(L, (0), axis=0)  # don't smooth first param at btm interface
    #L = np.concatenate((L,beta*np.eye(mlen)),axis=0)  # append ridge regr too
    return L



class InvTF(BaseEstimator):
    """ Frequentist inversion for weakly nonlinear problems using Tikhonov regularization.
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, fwd_deriv_code, minit, alpha=0.01, beta=50., max_iter=5,
                 dmtol=0.001, usefindiff=False, showplot=True, verbose=True):
        self.fwd_deriv_code = fwd_deriv_code
        self.minit = minit
        self.max_iter = max_iter  # only relevant after expanding this to nonlinear
        self.alpha = alpha
        self.beta = beta
        self.dmtol = dmtol  # only relevant after expanding this to nonlinear
        self.usefindiff = usefindiff
        self.showplot = showplot
        self.verbose = verbose
        #super().__init__(fit_intercept=False, copy_X=True)
        #super(InvT0, self).__init__(self, fit_intercept=False, copy_X=True)

    def fit(self, ymeas, mmeas=None, alg='optls'):   # alg: {'optls','mine'}
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
            # https://docs.scipy.org/doc/scipy/reference/optimize.html
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
            def fun(m):
                mlen = m.size
                L = create_findiff_mtx(mlen,self.beta)
                ypred,J = self.fwd_deriv_code(m)  # m: model params vector, J: derivs matrix
                resids = ymeas-ypred
                modelfunc = self.alpha * np.dot(L,m)
                modelfunc = modelfunc.reshape(len(modelfunc),1)
                f = np.squeeze(np.concatenate((resids,modelfunc),axis=0))
                return f

            def jac(m):
                mlen = m.size
                L = create_findiff_mtx(mlen,self.beta)
                ypred,J = self.fwd_deriv_code(m)  # m: model params vector, J: derivs matrix
                Jreg = self.alpha * L
                Jout = np.concatenate((J,Jreg))
                return Jout

            if self.usefindiff:
                jacfn='2-point'
            else:
                jacfn=jac
            if self.verbose:
                verblevel=2
            else:
                verblevel=0
            res = least_squares(fun, np.squeeze(self.minit), jac=jacfn,
                bounds=(0., 3.5), diff_step=None, verbose=verblevel, max_nfev=self.max_iter,
                method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0)
                #ftol=1e-4, xtol=1e-1, gtol=1e-8, x_scale=1.0)
                #ftol=1e0, xtol=1e-01, gtol=1e-01, x_scale=1.0)
                #ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0)

            if mmeas is not None:
                testMSE = cplxMSE(res.x.reshape(len(res.x),1),mmeas)
            else:
                testMSE = npl.nan
            ypred,J = self.fwd_deriv_code(res.x.reshape(len(res.x),1))
            ypred=np.log10(ypred)
            residnorm = norm(ypred-ymeas)
            print('resid norm',residnorm)
            L = create_findiff_mtx(len(self.minit),self.beta)
            print('maxeig JJ',np.real(np.amax(np.linalg.eigvals(np.dot(J.T,J)))))  # J'J has real eigvals but kept cplx type
            print('maxeig LL',np.amax(np.linalg.eigvals(np.dot(L.T,L))))
            if self.showplot:
                f, ax = plt.subplots(1, 2, figsize=(11,4))
                # plot the meas and pred data:
                # print('ypred',ypred)
                # print('ymeas',ymeas)
                ax[0].plot(ypred,'r.-')
                ax[0].plot(ymeas,'k.-')
                ax[0].grid()
                #ax[0].set_ylabel('cost')
                #ax[0].set_xlabel('iterations')
                ax[0].set_title('Measured (blk) and predicted (blu) data')
                # plot the init, true, and final model param vectors:
                ax[1].plot(self.minit,'g.-')
                ax[1].plot(res.x,'r.--')
                ax[1].plot(mmeas,'k.--')
                ax[1].grid()
                #ax[1].set_ylabel('model value')
                #ax[1].set_xlabel('indep var')
                ax[1].set_title('Model vectors (true=blk, init=grn, soln=red)')

            # return m,cost,misfit,modelnorm,norm(dm),testMSE
            return res.x,res.cost,np.nan,np.nan,np.nan,testMSE

        elif alg=='mine':
            cost = []
            m = self.minit
            mlen = len(m)
            if self.verbose:
                print('iter  alpha      cost       norm(dd)    norm(dm)   dmtol')
            for i in range(self.max_iter):
                ypred,X = self.fwd_deriv_code(m)  # m: model params vector, X: derivs matrix
                if self.usefindiff:
                    def tmpfwdcode(m):
                        return np.squeeze(self.fwd_deriv_code(m)[0])
                    X = jacfindiff(tmpfwdcode,m,dx=1.0e-6)  # dx=1.0e-6 is problem dependent!
                L = create_findiff_mtx(mlen,self.beta)
                G = np.concatenate((X, -self.alpha*L),axis=0)
                D = np.concatenate((ymeas-ypred, self.alpha*np.dot(L,m)),axis=0)
                misfit = cplxMSE(ymeas, ypred)
                modelnorm = norm(np.dot(L,m))**2
                current_cost = misfit + pow(self.alpha,2)*modelnorm
                dm,res,rnk,sv = lstsq(G,D)
                m = m + dm
                cost.append(current_cost)
                if self.verbose:
                    print('%3d  %6.1g  %10.3f  %10.3f  %10.2g  %6.3g' %
                        (i, self.alpha, current_cost, norm(ymeas-ypred), norm(dm), self.dmtol))
                if norm(dm) < self.dmtol:
                   break
            self.G = G
            self.ypred = ypred
            if mmeas is not None:
                testMSE = cplxMSE(m,mmeas)
            else:
                testMSE = npl.nan
            print('maxeig JJ',np.real(np.amax(np.linalg.eigvals(np.dot(X.T,X)))))  # X'X has real eigvals but kept cplx type
            print('maxeig LL',np.amax(np.linalg.eigvals(np.dot(L.T,L))))
            if self.showplot:
                f, ax = plt.subplots(1, 2, figsize=(11,4))
                # plot the cost (ie loss) per iterations:
                ax[0].semilogy(cost,'.-')  # (last element of cost)
                ax[0].grid()
                ax[0].set_ylabel('cost')
                ax[0].set_xlabel('iterations')
                ax[0].set_title('Cost history  (misfit^2 + alpha^2*modelnorm^2)')
                # plot the init, true, final, and evolution of model params:
                #print('m',np.squeeze(m.T))
                ax[1].plot(mmeas,'k')
                ax[1].plot(self.minit,'g')
                ax[1].plot(m,'r')
                ax[1].grid()
                #ax[1].set_ylabel('model value')
                ax[1].set_xlabel('indep var')
                ax[1].set_title('Model vectors')

        return m,cost[-1],misfit,modelnorm,norm(dm),testMSE


    def get_hyperparams(self):
        return (self.max_iter, self.dmtol, self.alpha)



class InvTB(BaseEstimator):
    """ Bayesian inversion for weakly nonlinear problem using Tikhonov regularization.
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, fwd_deriv_code, minit, mprior, Cprior, lb=-np.inf, ub=np.inf,
                 max_iter=5, dmtol=1e-8, diff_step=None, usefindiff=False, showplot=True, verbose=True):

        if Cprior.ndim==1:
            self.Cprior = np.diagflat(Cprior)
        else:
            self.Cprior = Cprior
        self.fwd_deriv_code = fwd_deriv_code
        self.minit = minit
        self.mprior = mprior
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter
        self.xtol = dmtol
        self.diff_step = diff_step
        self.usefindiff = usefindiff
        self.showplot = showplot
        self.verbose = verbose

    def fit(self, ymeas, ysigma, mmeas=None):
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

        if np.isscalar(ysigma):
            ysigma = ysigma*np.ones(len(ymeas))
        elif len(ysigma)!=len(ymeas):
            print('InvTB::fit() error:  len(ysigma)!=len(ymeas)')
        ysigma = ysigma.reshape(len(ysigma),1)  # ensure it's a Nx1 ndarray not a 1d array
        Dinvsqrt = np.diagflat(1.0/ysigma)  # (data inv covariance mtx)^(-1/2)
        Cinvsqrt = cholesky(inv(self.Cprior)).T  # (model inv covariance mtx)^(-1/2)

        def fun(m):
            mlen = m.size
            ypred,J = self.fwd_deriv_code(m)  # m: model params vector, J: derivs matrix
            resids = ymeas-ypred  #ypred-ymeas
            resids = np.dot(Dinvsqrt,resids)
            modelfunc = np.dot(Cinvsqrt,np.subtract(m,np.squeeze(self.mprior)))
            modelfunc = modelfunc.reshape(len(modelfunc),1)
            f = np.squeeze(np.concatenate((resids,modelfunc),axis=0))
            return f

        def jac(m):
            mlen = m.size
            ypred,J = self.fwd_deriv_code(m)  # m: model params vector, J: derivs matrix
            J = np.dot(Dinvsqrt,J)
            Jreg = Cinvsqrt
            Jout = np.concatenate((J,Jreg))
            return Jout

        if self.usefindiff:
            jacfn='2-point'
        else:
            jacfn=jac
        if self.verbose:
            verblevel=2
        else:
            verblevel=0

        res = least_squares(fun, np.squeeze(self.minit), jac=jacfn,
            bounds=(self.lb, self.ub), diff_step=self.diff_step, verbose=verblevel,
            max_nfev=self.max_iter, method='trf', ftol=1e-08, xtol=self.xtol, gtol=1e-08, x_scale=1.0)
        # https://docs.scipy.org/doc/scipy/reference/optimize.html
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        if mmeas is not None:
            testMSE = cplxMSE(res.x.reshape(len(res.x),1),mmeas)
        else:
            testMSE = npl.nan
        ypred,J = self.fwd_deriv_code(res.x.reshape(len(res.x),1))
        residnorm = norm(ypred-ymeas)
        #print('resid norm',residnorm)
        #print('maxeig JJ',np.real(np.amax(np.linalg.eigvals(np.dot(J.T,J)))))  # J'J has real eigvals but kept cplx type
        #print('maxeig Cinv',np.amax(np.linalg.eigvals(np.dot(Cinvsqrt.T,Cinvsqrt))))
        if self.showplot:
            f, ax = plt.subplots(1, 2, figsize=(11,4))
            # plot the meas and pred data:
            ax[0].plot(ypred,'r.-')
            ax[0].plot(ymeas,'k.-')
            ax[0].grid()
            ax[0].set_title('Measured (blk) and predicted (red) data')
            # plot the init, true, and final model param vectors:
            ax[1].plot(self.minit,'g.-')
            ax[1].plot(self.mprior,'c')
            ax[1].plot(res.x,'r.--')
            ax[1].plot(mmeas,'k.--')
            ax[1].grid()
            ax[1].set_title('Model vectors (meas=blk, pri=blu, ini=grn, sol=red)')

        return res.x,res.cost,np.nan,np.nan,np.nan,testMSE


    def get_hyperparams(self):
        return (self.max_iter, self.dmtol)
