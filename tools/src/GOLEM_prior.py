from castle.algorithms.gradient.notears.torch.golem import *
import numpy as np

class myGOLEM(GOLEM):
    def learn(self, data, columns=None, **kwargs):
        """
        Set up and run the GOLEM algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        X: numpy.ndarray
            [n, d] data matrix.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        lambda_1: float
            Coefficient of L1 penalty.
        lambda_2: float
            Coefficient of DAG penalty.
        equal_variances: bool
            Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter:int
            Number of iterations for training.
        learning_rate: float
            Learning rate of Adam optimizer. Default: 1e-3.
        seed: int
            Random seed. Default: 1.
        checkpoint_iter: int
            Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        B_init: numpy.ndarray or None
            [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.
        """

        X = Tensor(data, columns=columns)
        
        causal_matrix = self._golem(X)
        self.causal_matrix = Tensor(causal_matrix, index=X.columns,
                                    columns=X.columns)
        
    def _golem(self, X):
        """
        Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

        Parameters
        ----------
        X: numpy.ndarray
            [n, d] data matrix.
        
        Return
        ------
        B_result: np.ndarray
            [d, d] estimated weighted matrix.
        
        Hyperparameters
        ---------------
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
        """
        set_seed(self.seed)
        n, d = X.shape
        X = torch.Tensor(X).to(self.device)

        # Set up model
        model = GolemModel(n=n, d=d, lambda_1=self.lambda_1,
                           lambda_2=self.lambda_2,
                           equal_variances=self.equal_variances,
                           B_init=self.B_init,
                           device=self.device)

        self.train_op = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        logging.info("Started training for {} iterations.".format(int(self.num_iter)))
        for i in range(0, int(self.num_iter) + 1):
            model(X)
            score, likelihood, h, B_est = model.score, model.likelihood, model.h, model.B
            
            if i > 0:  # Do not train here, only perform evaluation
                # Optimizer
                self.loss = score+_prior(B_est, self.w_prior, self.prob_prior)
                self.train_op.zero_grad()
                self.loss.backward()
                self.train_op.step()

            if self.checkpoint_iter is not None and i % self.checkpoint_iter == 0:
                logging.info("[Iter {}] score={:.3f}, likelihood={:.3f}, h={:.1e}".format( \
                    i, score, likelihood, h))

        # Post-process estimated solution and compute results
        B_processed = postprocess(B_est.cpu().detach().numpy(), graph_thres=0.3)
        B_result = (B_processed != 0).astype(int)

        return B_result

    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=w_prior
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.lambda_1=lambda1*2

def _prior(W, w_prior=None, prob_prior=0):
            
    if w_prior is None:
        return 0, np.zeros_like(W)
    # Edge existence where w_prior is 1, forbidden where it's -1, do nothing where it's 0
    w_prior = torch.from_numpy(w_prior)
    W_b = torch.abs(2*torch.sigmoid(W)-1)
    prob_exist = W_b * prob_prior + (1-W_b) * (1-prob_prior)
    prob_forb  = (1-W_b) * prob_prior + W_b * (1-prob_prior)
    prior = torch.sum(torch.log(prob_exist[w_prior == 1])) + \
        torch.sum(torch.log(prob_forb[w_prior == -1]))
    prior = - prior
    
    return prior

