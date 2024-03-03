import numpy as np
from numpy import array,newaxis,ones,zeros,shape
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm,det,inv,pinv # scipy.linalg is a bit more full-features, includes additional routines, 
                                            # and subsumes numpy.linalg.  It's generally its better to use scipy.linalg.
import numpy.random as random

def plogdeth(A):
    """For a full-rank Hermitian matrix, return log(det(A)).  If not full-rank, ignore zero eigenvalues."""
    v = eigvalsh(A)
    return np.sum(np.log(v[~np.isclose(v,0)]))
    # first taking the log and then summing is safer and more numerically satble and acurate, as it avoids really big numbers

class Classifier:
    def predict_log_proba(self,X):
        raise NotImplementedError
    def predict(self,X):
        """Returns Bayes Optimal predicted labels"""
        return np.array(self.labels)[np.argmax(self.predict_log_proba(X),1)]
    def score(self,X,y):
        """Accuracy rate"""
        return np.mean(self.predict(X)==y)
    def error(self,X,y):
        """Error rate (empirical 0/1 error on the give data)"""
        return 1-self.score(X,y)

def homogeneous_phi(x):
    return x
def affine_phi(x):
    return np.hstack( ( np.ones( (np.size(x,0),1) ) , x ) )
def quad_phi(x):
    return np.hstack( ( np.ones( (np.size(x,0),1) ) , x , x[:,newaxis,:]*x[:,:,newaxis]) )

class LinearClassifier(Classifier):
    def __init__(self,phi=affine_phi):
        self.phi=phi
    def predict(self,X):
        return np.sign(self.phi(X)@self.w)

class GaussianMix(Classifier):
    """Represents a mixture of Gaussians, with one label per Gaussian."""
    def __init__(self,mus=[np.array([-1,0]),np.array([1,0])],Sigmas=1,ps=None,labels=[-1,1]):
            """Instantiate the mixture based on specified parameters"""
            from numbers import Number
            d = len(mus[0]) # the dimensionality 
            if isinstance(Sigmas,Number): # A scalar can be passed for Sigmas, indicating a scaled identity covariance
                Sigmas = Sigmas*np.eye(d)
            if isinstance(Sigmas,np.ndarray): # If only a single matrix is passed, it is used as the covariance for all classes
                Sigmas = [Sigmas]*len(labels)
            if ps is None: # default to uniform over the components
                ps = np.array([1/len(labels)]*len(labels))
            self.d = d
            self.mus=mus
            self.Sigmas=Sigmas
            self.ps=ps
            self.labels=labels
    def generate(self,m):
        """Generate m samples from the mixture"""
        y = random.choice(self.labels,p=self.ps,size=m)
        x = np.empty((m,self.d))
        for label,mu,Sigma in zip(self.labels,self.mus,self.Sigmas):
            thism = sum(y==label)
            x[y==label] = random.randn(thism,self.d)@sqrtm(Sigma)+mu
        return x,y
    
    def predict_log_proba(self,x):
        """Log posteriors.  
        Returns logp, where logp[i,j]=log P(labels[j]|x[i,:])"""
        return np.stack([np.log(p) -0.5*np.sum(((x-mu)@pinv(Sigma))*(x-mu),1) -0.5*plogdeth(2*np.pi*Sigma) 
                             for p,mu,Sigma in zip(self.ps,self.mus,self.Sigmas)]).T
    def lin_logpost(self):
        """returns b,w, such that log P(labels[i]|x) \propto b[i] + w[i,:]@x"""
        assert allallclose(self.Sigmas)
        Sinv=pinv(self.Sigmas[0])
        mus = array(self.mus)
        b = np.log(self.ps) - 0.5 * sum(mus[:,newaxis,:]*mus[:,:,newaxis]*Sinv,axis=(1,2))
        w = mus@Sinv
        return b,w
    def quad_logpost(self):
        """returns b,w,H such that log P(labels[i]|x)=b[i] + w[i]@x + 0.5*x@H[i]@x"""
        Sinvs = [ pinv(Sigma) for Sigma in self.Sigmas ]
        b = [ np.log(p) - 0.5*mu@Sinv@mu +0.5*plogdeth(Sinv/(2*np.pi)) for p,mu,Sinv in zip(self.ps,self.mus,Sinvs)]
        w = [ mu@Sinv for mu,Sinv in zip(self.mus,Sinvs) ]
        H = [ -S for S in Sinvs]
        return b,w,H
    def lin_pred(self):
        """For two labels, return b,w such the discriminant is b+w@x"""
        bs,ws = self.lin_logpost()
        return bs[1]-bs[0],ws[1]-ws[0]
    def quad_pred(self):
        """For two labels, returns b,w,H such that the discriminant is b+w@x+x@H@x"""
        bs,ws,Hs = self.quad_logpost()
        return bs[1]-bs[0],ws[1]-ws[0],Hs[1]-Hs[0]
    def as_linear_predictor(self):
        """Returns a Linear Predictor (as a LinearClassifier instance) corresponding to the
        Bayes classifier for the mixture, if it is indeed linear."""
        clf = LinearClassifier()
        b,w = self.lin_pref()
        clf.w = np.block([b,w])
        return clf
    def as_quad_pred(self):
        """Returns a Quadratic Predctor (as a LinearClassifier instance over quadratic features)
        corresponding to the Bayes classifier for the mixture."""
        clf = LinearClassifier(phi=quad_phi)
        b,w,H = self.equad_pred()
        clf.w = np.block([b,w,np.flatten(H)])
        return clf

class LearnedGaussianMix(GaussianMix):
    def __init__(self,labels=None):
        """Create a "blank" Gaussian Mixture, without parameters yet (and hence can't be used).
        Use self.fit(X,y) to fit the parameters to data"""
        self.labels=labels
    def fit(self,Xs,ys):
        if self.labels is None:
            self.labels = np.unique(ys)
        labels = self.labels
        ps=np.mean(ys[:,newaxis]==labels,0)
        mus = [ np.mean(Xs[ys==y,:],0) for y in labels]
        Sigmas = [ np.cov(Xs[ys==y].T,bias=True) for y in labels]
        GaussianMix.__init__(self,mus=mus,Sigmas=Sigmas,ps=ps,labels=labels) 
class LDA(LearnedGaussianMix):
    """Covariances matrices constrained to be the same"""
    def fit(self,*args,**kwargs):
        super().fit(*args,**kwargs)
        self.Sigmas = [ np.sum( np.array(self.Sigmas) * self.ps[:,newaxis,newaxis],0) ] * len(self.labels)
class DiagGM(LearnedGaussianMix):
    """Covariances matrices constrained to be diagonal"""
    def fit(self,*args,**kwargs):
        super().fit(*args,**kwargs)
        self.Sigmas=[ np.diag(np.diag(Sigma)) for Sigma in self.Sigmas]
class DiagLDA(LDA,DiagGM):
    """Covariance matrices constrained to be the same diagonal matrix"""
    pass
class SphericalGM(DiagGM):
    """Covariance matrices constrained to be spherical"""
    def fit(self,*args,**kwargs):
        super().fit(*args,**kwargs)
        self.Sigmas=[ np.mean(np.diag(Sigma))*np.eye(self.d) for Sigma in self.Sigmas]
class SphericalLDA(LDA,SphericalGM):
    """Covariance matrices constrained to be the same spherical"""
    pass
class UnitSphericalGM(SphericalLDA):
    """Covariance matrices constrained to be spherical with unit variance, i.e. identity"""
    def fit(self,*args,**kwargs):
        super().fit(*args,**kwargs)
        # going through the LDA and Spherical, or even calculating the covariance, is redundant,
        # but this is subclassed to indicate it has all the properties of them
        self.Sigmas=[ np.eye(self.d) ]*len(self.labels)

class ERMLinearClassifier(LinearClassifier):
    """Linear predictor trained by minimizing the (regularized) empirical risk, with respect to some loss function.
    
    This is a base class that does not define which loss function and which regularizer is used---these need to be
    defined in subclasses.  The loss function is defined in terms of a scalar function loss(z) where 
    ell(prediction,y)=loss(y*prediction).
    
    In particular, in order to be used, base classes must define the following methods:
    loss_func(self,z) -> array of the same shape as z, with values loss(z)
    loss_derivative(self,z) -> array the same shape as z, with values loss'(z).
    
    The regularizer is defined via the methods regularizer(self,w) and its gradient regularizer_grad(self,w).
    In the based class, the regularizer is zero (no regularization).  It can be optionally implemented in a 
    base class."""
    def __init__(self,lmb=0,*args,**kwargs):
        """lmb is the regularization tradeoff parameters.  Default lmb=0 for no regularization."""
        super().__init__()
        self.lmb=lmb
    def fit(self,X,y,verbose=False):
        """Fit the linear predictor by minimizing the empirical risk"""
        from scipy.optimize import minimize
        yphiX = y[:,newaxis]*self.phi(X)
        m,d = np.shape(yphiX)
        def training_obj(w):
            # training objective, for the given data, as a function of a predictor w
            return np.mean(self.loss_func(yphiX@w)) + self.lmb * self.regularizer(w)
        def training_grad(w):
            # gradient of training_obj with respect to w
            return self.loss_derivative(yphiX@w)@yphiX/m + self.lmb * self.regularizer_grad(w)
            # you will want to use self.loss_derivative and self.regularizer_grad
        minimization = minimize(training_obj, zeros(d), jac=training_grad)
        self.w = minimization.x
    def regularizer(self,w):
        return 0
    def regularizer_grad(self,w):
        return np.zeros(np.shape(w))  # warning! changed since initial implementation
    
    
class HingeReg(ERMLinearClassifier):
    def loss_func(self,z):
        return np.maximum(0,1-z)
    def loss_derivative(self,z):
        return -1*(z<1)
class LogisticReg(ERMLinearClassifier):
    def loss_func(self,z):
         return np.log(1+np.exp(-z))
    def loss_derivative(self,z):
         return -1/(1+np.exp(z))
class L2RegLinear(ERMLinearClassifier):
    def regularizer(self,w):
        return np.sum(w**2)/2
    def regularizer_grad(self,w):
        return w
class L1RegLinear(ERMLinearClassifier):
    def regularizer(self,w):
        return np.sum(np.abs(w))
    def regularizer_grad(self,w):
        return np.sign(w)
class LogisticL2Reg(LogisticReg,L2RegLinear):
    pass
class HingeL2Reg(HingeReg,L2RegLinear):
    pass
class LogisticL1Reg(LogisticReg,L1RegLinear):
    pass
class HingeL1Reg(HingeReg,L1RegLinear):
    pass

class NoMoreFeatureToSelect(StopIteration):
    pass

class ForwardGreedySelection:
    """ForwardGreedySelection wrapper.

    Performs forward greedy selection for any given learning rule,
    adding the feature that reduced either the training error or a
    validation error (or score function given by the predictor).

    After fitting:
        self.support is a list of the selected features, in the order
        in which they were selected.
        self.h is the predictor on those features, as an instance of
        learningRule."""
    def __init__(self, learningRule, target_supp_size, **learningRule_kwargs):
        """learningRule should be a class definint the following
        methods:
            learningRule.fit(X,y)
            learningRule.predict(X)
            learningRule.score(X,y) -- higher is better (eg accuracy)
        the following methods are also supported:
            predict_log_proba, score, error."""
        self.learningRule = learningRule
        self.learningRule_kwargs = learningRule_kwargs
        self.target_supp_size = target_supp_size
        self.support = []
        self.supp_size = 0
    def greedy_add_feature(self,X,y,Xval=None,yval=None,**kwargs):
        """Add one additional feature to current support"""
        support=self.support
        if Xval is None:
            # minimize empirical error rather than validation error
            Xval = X
            yval = y
        best_score = float('-inf')
        for i in range(np.size(X,1)):
            if i not in support:
                h = self.learningRule(**self.learningRule_kwargs)
                h.fit(X[:,support+[i]],y,**kwargs)
                score = h.score(Xval[:,support+[i]],yval)
                if score>best_score:
                    best_score = score
                    best_i = i
                    best_h = h
        if best_score == float('-inf'):
            raise NoMoreFeatureToSelect
        self.support=support+[best_i]
        self.h = best_h
        self.supp_size = self.supp_size+1
        assert self.supp_size == len(self.support)
        return self
    def fit(self,*args,**kwargs):
        """fit(X,y,Xval,yval,....).  Fit from empyt support.
        Selects self.target_supp_size features.
        Xval,yval can be omited, in which case the training error (on
        X,y) is used.
        Any additional keyword arguments are passed learningRule.fit()"""
        self.support = []
        self.supp_size = 0
        self.fit_more(*args,**kwargs)
    def fit_more(self,*args,**kwargs):
        "Continue fitting from current support"
        try:
            while self.supp_size < self.target_supp_size:
                self.greedy_add_feature(*args,**kwargs)
        except NoMoreFeatureToSelect:
            # less than k fetures total, but that's OK
            pass
    def transform(self,X):
        "Returns a data set with only the selected fetures of X"
        return X[:,self.support]
    def predict(self,X):
        return self.h.predict(self.transform(X))
    def predict_log_proba(self,X):
        return self.h.predict_log_proba(self.transform(X))
    def score(self,X,*args,**kwargs):
        return self.h.score(self.transform(X),*args,**kwargs)
    def error(self,X,*args,**kwargs):
        return self.h.error(self.transform(X),*args,**kwargs)

    # The following is not used, but is included in case you want to experiment with backward selection
    def greedy_remove_feature(self,X,y,Xval=None,yval=None):
        support=self.support
        if Xval is None:
            # minimize empirical error rather than validation error
            Xval = X
            yval = y
        best_score = float('-inf')
        for i in range(len(support)):
            h = self.learningRule(**self.learningRule_kwargs)
            h.fit(X[:,support[:i]+support[i+1:]],y)
            score = h.score(Xval[:,support[:i]+support[i+1:]],yval)
            if score>best_score:
                best_score = score
                best_i = i
                best_h = h
        if best_score == float('-inf'):
            raise NoMoreFeatureToSelect
        self.support=support[:best_i]+support[best_i+1:]
        self.h = best_h
        self.supp_size = self.supp_size-1
        assert self.supp_size == len(self.support)
        return self
    
            
