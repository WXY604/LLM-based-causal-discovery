from castle.algorithms.gradient.gae.torch.trainers.al_trainer import *
from castle.algorithms.gradient.gae.torch.gae import *

class myGAE(GAE):
    def learn(self, data, columns=None, **kwargs):

        x = data

        self.n, self.d = x.shape[:2]
        if x.ndim == 2:
            x = x.reshape((self.n, self.d, 1))
            self.input_dim = 1
        elif x.ndim == 3:
            self.input_dim = x.shape[2]

        w_est = self._gae(x).detach().cpu().numpy()

        self.weight_causal_matrix = Tensor(w_est,
                                           index=columns,
                                           columns=columns)
        causal_matrix = (abs(w_est) > self.graph_thresh).astype(int)
        self.causal_matrix = Tensor(causal_matrix,
                                    index=columns,
                                    columns=columns)
    
    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=w_prior
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.penalty_lambda=lambda1*0.00

    def _gae(self, x):
        
        set_seed(self.seed)
        model = AutoEncoder(d=self.d,
                            input_dim=self.input_dim,
                            hidden_layers=self.hidden_layers,
                            hidden_dim=self.hidden_dim,
                            activation=self.activation,
                            device=self.device,
                            )
        trainer = my_ALTrainer(n=self.n,
                            d=self.d,
                            model=model,
                            lr=self.lr,
                            init_iter=self.init_iter,
                            alpha=self.alpha,
                            beta=self.beta,
                            rho=self.init_rho,
                            l1_penalty=self.penalty_lambda,
                            rho_thresh=self.rho_thresh,
                            h_thresh=self.h_thresh,  # 1e-8
                            early_stopping=self.early_stopping,
                            early_stopping_thresh=self.early_stopping_thresh,
                            gamma=self.gamma,
                            seed=self.seed,
                            device=self.device)
        trainer.load_prior(self.w_prior,self.prob_prior)
        
        w_est = trainer.train(x=x,
                              epochs=self.epochs,
                              update_freq=self.update_freq)
        w_est = w_est / torch.max(abs(w_est))

        return w_est

class my_ALTrainer(ALTrainer):
    def train_step(self, x, update_freq, alpha, rho):

        curr_mse, curr_h, w_adj = None, None, None
        for _ in range(update_freq):
            torch.manual_seed(self.seed)
            curr_mse, w_adj = self.model(x)
            curr_h = compute_h(w_adj)
            loss = ((0.5 / self.n) * curr_mse
                    + self.l1_penalty * torch.norm(w_adj, p=1)
                    + alpha * curr_h + 0.5 * rho * curr_h * curr_h
                    + _prior(w_adj,self.w_prior,self.prob_prior))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if _ % LOG_FREQUENCY == 0:
                logging.info(f'Current loss in step {_}: {loss.detach()}')

        return curr_mse, curr_h, w_adj
    
    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=w_prior
        self.prob_prior=prob_prior


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
    
    return prior *0.3
