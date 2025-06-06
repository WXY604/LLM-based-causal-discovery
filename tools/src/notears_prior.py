from castle.algorithms.gradient.notears.linear import *
import torch

class myNotears_prior(Notears):
    def __init__(self, lambda1=0.1,
                 sigma=1.0, 
                 loss_type='l2', 
                 max_iter=100, 
                 h_tol=1e-8, 
                 rho_max=1e+16, 
                 w_threshold=0.3):

        super().__init__()

        self.lambda1 = lambda1
        self.sigma = sigma
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        
        self.w_prior=None
        self.prob_prior=0

    def learn(self, data, columns=None, **kwargs):
        """
        Set up and run the Notears algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """
        #X = Tensor(data, columns=columns)
        #用tensor似乎会慢4倍？
        X=data
        W_est = self.notears_linear(X, lambda1=self.lambda1,
                                    sigma=self.sigma, 
                                    loss_type=self.loss_type,
                                    max_iter=self.max_iter, 
                                    h_tol=self.h_tol, 
                                    rho_max=self.rho_max,
                                    w_threshold=self.w_threshold)
        causal_matrix = (abs(W_est) > self.w_threshold).astype(int)
        X = Tensor(data, columns=columns)
        self.weight_causal_matrix = Tensor(W_est,
                                           index=X.columns,
                                           columns=X.columns)
        self.causal_matrix = Tensor(causal_matrix, index=X.columns,
                                    columns=X.columns)
        
    def notears_linear(self,X, lambda1=0.1, loss_type='pdf', max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3,sigma=1.0):
        """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

        Args:
            X (np.ndarray): [n, d] sample matrix
            lambda1 (float): l1 penalty parameter
            loss_type (str): l2, logistic, poisson
            max_iter (int): max num of dual ascent steps
            h_tol (float): exit if |h(w_est)| <= htol
            rho_max (float): exit if rho >= rho_max
            w_threshold (float): drop edge if |weight| < threshold

        Returns:
            W_est (np.ndarray): [d, d] estimated DAG
        """
        def _loss(W, sigma=sigma):
            """Evaluate value and gradient of loss."""
            M = X @ W
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[0] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[0] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[0] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
            elif loss_type == 'pdf':
                R = X - M
                loss= 0.5 / X.shape[0] * (R ** 2).sum() / sigma**2
                G_loss = - 1.0 / X.shape[0] * X.T @ R / sigma**2
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            # E = slin.expm(W * W)  # (Zheng et al. 2018)
            # h = np.trace(E) - d
            # A different formulation, slightly faster at the cost of numerical stability
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            
            return h, G_h
        
        def _prior(W, w_prior=None, prob_prior=0):
            
            if w_prior is None:
                return 0, np.zeros_like(W)
            # Edge existence where w_prior is 1, forbidden where it's -1, do nothing where it's 0
            W = torch.from_numpy(W)
            w_prior = torch.from_numpy(w_prior)
            W.requires_grad = True
            W_b = torch.abs(2*torch.sigmoid(W)-1)
            prob_exist = W_b * prob_prior + (1-W_b) * (1-prob_prior)
            prob_forb  = (1-W_b) * prob_prior + W_b * (1-prob_prior)
            prior = torch.sum(torch.log(prob_exist[w_prior == 1])) + \
                torch.sum(torch.log(prob_forb[w_prior == -1]))
            prior = - prior
            
            prior=prior*0.4 #Replace the role of sigma
            prior.backward()
            G_prior = W.grad
            return prior.detach().numpy(), G_prior.detach().numpy()

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = _loss(W)

            h, G_h = _h(W)
            prior, G_prior = _prior(W,self.w_prior, self.prob_prior) 
            G_prior[(W<0.1) & (W>-0.1)]=G_prior[(W<0.1) & (W>-0.1)]*0.3
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum() + prior
            G_smooth = G_loss  + (rho * h + alpha) * G_h  + G_prior
            # g_l1 = np.where(W>0, 1, -1)
            g_obj = np.concatenate((G_smooth + lambda1 , - G_smooth + lambda1 ), axis=None)
            return obj, g_obj

        n, d = X.shape
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        if loss_type == 'l2' or loss_type == 'pdf':
            X = X - np.mean(X, axis=0, keepdims=True)
        for i in range(max_iter):
            # print(f"iter: {i} rho: {rho:.4f}, alpha: {rho:.4f}, w_est: {np.abs(w_est).sum()}, h: {h:.4f}")
            w_new, h_new = None, None
            while rho < rho_max:
                # print(f"rho: {rho:.4f}, alpha: {rho:.4f}, w_est: {np.abs(w_est).sum()}, h: {h:.4f}")
                sol = sopt.minimize(lambda w: _func(w), w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break
        W_est = _adj(w_est)
        W_est[np.abs(W_est) < w_threshold] = 0
        return W_est
    
    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=w_prior
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.penalty_lambda=lambda1