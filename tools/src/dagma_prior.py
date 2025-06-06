import numpy as np
from dagma.linear import *
import typing
from tqdm.auto import tqdm
import copy

class mydagma_prior(DagmaLinear):
    def __init__(self, loss_type: str, verbose: bool = False, dtype: type = np.float64) -> None:
        r"""
        Parameters
        ----------
        loss_type : str
            One of ["l2", "logistic"]. ``l2`` refers to the least squares loss, while ``logistic``
            refers to the logistic loss. For continuous data: use ``l2``. For discrete 0/1 data: use ``logistic``.
        verbose : bool, optional
            If true, the loss/score and h values will print to stdout every ``checkpoint`` iterations,
            as defined in :py:meth:`~dagma.linear.DagmaLinear.fit`. Defaults to ``False``.
        dtype : type, optional
           Defines the float precision, for large number of nodes it is recommened to use ``np.float64``. 
           Defaults to ``np.float64``.
        """
        super().__init__(loss_type,verbose,dtype)
        losses = ['l2', 'logistic']
        assert loss_type in losses, f"loss_type should be one of {losses}"
        self.loss_type = loss_type
        self.dtype = dtype
        self.vprint = print if verbose else lambda *a, **k: None

    def learn(self, 
            X: np.ndarray,
            lambda1: float = 0.03, 
            w_threshold: float = 0.3, 
            T: int = 5,
            mu_init: float = 1.0, 
            mu_factor: float = 0.1, 
            s: typing.Union[typing.List[float], float] = [1.0, .9, .8, .7, .6], 
            warm_iter: int = 3e4, 
            max_iter: int = 6e4, 
            lr: float = 0.0003, 
            checkpoint: int = 1000, 
            beta_1: float = 0.99, 
            beta_2: float = 0.999,
            exclude_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None, 
            include_edges: typing.Optional[typing.List[typing.Tuple[int, int]]] = None,
        ) -> np.ndarray :
        r"""
        Runs the DAGMA algorithm and returns a weighted adjacency matrix.

        Parameters
        ----------
        X : np.ndarray
            :math:`(n,d)` dataset.
        lambda1 : float
            Coefficient of the L1 penalty. Defaults to 0.03.
        w_threshold : float, optional
            Removes edges with weight value less than the given threshold. Defaults to 0.3.
        T : int, optional
            Number of DAGMA iterations. Defaults to 5.
        mu_init : float, optional
            Initial value of :math:`\mu`. Defaults to 1.0.
        mu_factor : float, optional
            Decay factor for :math:`\mu`. Defaults to 0.1.
        s : typing.Union[typing.List[float], float], optional
            Controls the domain of M-matrices. Defaults to [1.0, .9, .8, .7, .6].
        warm_iter : int, optional
            Number of iterations for :py:meth:`~dagma.linear.DagmaLinear.minimize` for :math:`t < T`. Defaults to 3e4.
        max_iter : int, optional
            Number of iterations for :py:meth:`~dagma.linear.DagmaLinear.minimize` for :math:`t = T`. Defaults to 6e4.
        lr : float, optional
            Learning rate. Defaults to 0.0003.
        checkpoint : int, optional
            If ``verbose`` is ``True``, then prints to stdout every ``checkpoint`` iterations. Defaults to 1000.
        beta_1 : float, optional
            Adam hyperparameter. Defaults to 0.99.
        beta_2 : float, optional
            Adam hyperparameter. Defaults to 0.999.
        exclude_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
            Tuple of edges that should be excluded from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.
        include_edges : typing.Optional[typing.List[typing.Tuple[int, int]]], optional
            Tuple of edges that should be included from the DAG solution, e.g., ``((1,3), (2,4), (5,1))``. Defaults to None.

        Returns
        -------
        np.ndarray
            Estimated DAG from data.
        
        
        .. important::

            If the output of :py:meth:`~dagma.linear.DagmaLinear.fit` is not a DAG, then the user should try larger values of ``T`` (e.g., 6, 7, or 8) 
            before raising an issue in github.
        
        .. warning::
            
            While DAGMA ensures to exclude the edges given in ``exclude_edges``, the current implementation does not guarantee that all edges
            in ``included edges`` will be part of the final DAG.
        """ 
        
        ## INITALIZING VARIABLES 
        self.X, self.checkpoint = X, checkpoint
        self.lambda1 = self.penalty_lambda
        self.n, self.d = X.shape
        self.Id = np.eye(self.d).astype(self.dtype)
        
        if self.loss_type == 'l2':
            self.X -= X.mean(axis=0, keepdims=True)
        
        self.exc_r, self.exc_c = None, None
        if not hasattr(self,'inc_r'):
            self.inc_r=None
        if not hasattr(self,'inc_c'):
            self.inc_c=None
        
        if exclude_edges is not None:
            if type(exclude_edges) is tuple and type(exclude_edges[0]) is tuple and np.all(np.array([len(e) for e in exclude_edges]) == 2):
                self.exc_r, self.exc_c = zip(*exclude_edges)
            else:
                ValueError("blacklist should be a tuple of edges, e.g., ((1,2), (2,3))")
        
        if include_edges is not None:
            if type(include_edges) is tuple and type(include_edges[0]) is tuple and np.all(np.array([len(e) for e in include_edges]) == 2):
                self.inc_r, self.inc_c = zip(*include_edges)
            else:
                ValueError("whitelist should be a tuple of edges, e.g., ((1,2), (2,3))")        
            
        self.cov = X.T @ X / float(self.n)    
        self.W_est = np.zeros((self.d,self.d)).astype(self.dtype) # init W0 at zero matrix
        mu = mu_init
        if type(s) == list:
            if len(s) < T: 
                self.vprint(f"Length of s is {len(s)}, using last value in s for iteration t >= {len(s)}")
                s = s + (T - len(s)) * [s[-1]]
        elif type(s) in [int, float]:
            s = T * [s]
        else:
            ValueError("s should be a list, int, or float.")    
        
        ## START DAGMA
        with tqdm(total=(T-1)*warm_iter+max_iter) as pbar:
            for i in range(int(T)):
                self.vprint(f'\nIteration -- {i+1}:')
                lr_adam, success = lr, False
                inner_iters = int(max_iter) if i == T - 1 else int(warm_iter)
                while success is False:
                    W_temp, success = self.minimize(self.W_est.copy(), mu, inner_iters, s[i], lr=lr_adam, beta_1=beta_1, beta_2=beta_2, pbar=pbar)
                    if success is False:
                        self.vprint(f'Retrying with larger s')
                        lr_adam *= 0.5
                        s[i] += 0.1
                self.W_est = W_temp
                mu *= mu_factor
        
        ## Store final h and score values and threshold
        self.h_final, _ = self._h(self.W_est)
        self.score_final, _ = self._score(self.W_est)
        
        self.weight_causal_matrix=copy.copy(self.W_est)
        self.W_est[np.abs(self.W_est) < w_threshold] = 0
        self.causal_matrix=copy.copy(self.W_est!=0).astype(int)
        return self.W_est

    def load_prior(self,w_prior=None, prob_prior=0):
        self.inc_r, self.inc_c =np.where(w_prior==1)
        #self.w_prior=w_prior
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.penalty_lambda=lambda1