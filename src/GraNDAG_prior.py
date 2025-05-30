from castle.algorithms.gradient.gran_dag.torch.gran_dag import *

class myGraNDAG(GraNDAG):
    
    def learn(self, data, columns=None, **kwargs):
        """Set up and run the Gran-DAG algorithm

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            include Tensor.data
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """

        # Control as much randomness as possible
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Use gpu
        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.precision:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            if self.precision:
                torch.set_default_tensor_type('torch.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.DoubleTensor')
            device = torch.device('cpu')
        self.device = device

        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        if data.shape[1] != self.input_dim:
            raise ValueError("The number of variables is `{}`, "
                             "the param input_dim is `{}`, "
                             "they must be consistent"
                             ".".format(data.shape[1], self.input_dim))

        if self.model_name == "NonLinGauss":
            self.model = NonlinearGauss(input_dim=self.input_dim,
                                        hidden_num=self.hidden_num,
                                        hidden_dim=self.hidden_dim,
                                        output_dim=2,
                                        nonlinear=self.nonlinear,
                                        norm_prod=self.norm_prod,
                                        square_prod=self.square_prod)
        elif self.model_name == "NonLinGaussANM":
            self.model = my_NonlinearGaussANM(input_dim=self.input_dim,
                                           hidden_num=self.hidden_num,
                                           hidden_dim=self.hidden_dim,
                                           output_dim=1,
                                           nonlinear=self.nonlinear,
                                           norm_prod=self.norm_prod,
                                           square_prod=self.square_prod)
        else:
            raise ValueError(
                "self.model has to be in {NonLinGauss, NonLinGaussANM}")

        # create NormalizationData
        train_data = NormalizationData(data, train=True,
                                       normalize=self.normalize)
        test_data = NormalizationData(data, train=False,
                                      normalize=self.normalize,
                                      mean=train_data.mean,
                                      std=train_data.std)

        # apply preliminary neighborhood selection if input_dim > 50
        if self.use_pns:
            if self.num_neighbors is None:
                num_neighbors = self.input_dim
            else:
                num_neighbors = self.num_neighbors

            self.model = neighbors_selection(model=self.model, all_samples=data,
                                             num_neighbors=num_neighbors,
                                             thresh=self.pns_thresh)

        # update self.model by train
        self._train(train_data=train_data, test_data=test_data)

        # update self.model by run _to_dag
        self._to_dag(train_data)

        self._causal_matrix = Tensor(self.model.adjacency.detach().cpu().numpy(),
                                     index=data.columns,
                                     columns=data.columns)
        
    def _train(self, train_data, test_data):
        """
        Applying augmented Lagrangian to solve the continuous constrained problem.

        Parameters
        ----------
        train_data: NormalizationData
            train samples
        test_data: NormalizationData object
            test samples for validation
        """

        # initialize stuff for learning loop
        aug_lagrangians = []
        aug_lagrangian_ma = [0.0] * (self.iterations + 1)
        aug_lagrangians_val = []
        grad_norms = []
        grad_norm_ma = [0.0] * (self.iterations + 1)

        w_adjs = np.zeros((self.iterations,
                           self.input_dim,
                           self.input_dim), dtype=np.float32)

        hs = []
        not_nlls = []  # Augmented Lagrangian minus (pseudo) NLL
        nlls = []  # NLL on train
        nlls_val = []  # NLL on validation

        # Augmented Lagrangian stuff
        mu = self.mu_init
        lamb = self.lambda_init
        mus = []
        lambdas = []

        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not implemented"
                                      .format(self.optimizer))

        # Learning loop:
        for iter in tqdm(range(self.iterations), desc='Training Iterations'):
            # compute loss
            self.model.train()
            x, _ = train_data.sample(min(self.batch_size, train_data.n_samples))
            # Initialize weights and bias
            weights, biases, extra_params = self.model.get_parameters(mode="wbx")
            loss = - torch.mean(
                self.model.compute_log_likelihood(x, weights, biases, extra_params))
            nlls.append(loss.item())
            self.model.eval()

            # constraint related
            w_adj = self.model.get_w_adj()
            h = compute_constraint(self.model, w_adj)

            # compute augmented Lagrangian
            aug_lagrangian = loss + 0.5 * mu * h ** 2 + lamb * h + _prior(w_adj,self.w_prior, self.prob_prior) + self.l1_penalty * torch.norm(w_adj, p=1)
            # optimization step on augmented lagrangian
            optimizer.zero_grad()
            aug_lagrangian.backward()
            optimizer.step()

            # clamp edges
            if self.edge_clamp_range != 0:
                with torch.no_grad():
                    to_keep = (abs(w_adj) > self.edge_clamp_range) * 1
                    self.model.adjacency *= to_keep

            # logging
            w_adjs[iter, :, :] = w_adj.detach().cpu().numpy().astype(np.float32)
            mus.append(mu)
            lambdas.append(lamb)
            not_nlls.append(0.5 * mu * h.item() ** 2 + lamb * h.item())

            # compute augmented lagrangian moving average
            aug_lagrangians.append(aug_lagrangian.item())
            aug_lagrangian_ma[iter + 1] = aug_lagrangian_ma[iter] + \
                                          0.01 * (aug_lagrangian.item() -
                                                  aug_lagrangian_ma[iter])
            grad_norms.append(self.model.get_grad_norm("wbx").item())
            grad_norm_ma[iter + 1] = grad_norm_ma[iter] + \
                                     0.01 * (grad_norms[-1] - grad_norm_ma[iter])

            # compute loss on whole validation set
            if iter % self.stop_crit_win == 0:
                with torch.no_grad():
                    x, _ = test_data.sample(test_data.n_samples)
                    loss_val = - torch.mean(self.model.compute_log_likelihood(x,
                                                                 weights,
                                                                 biases,
                                                                 extra_params))
                    nlls_val.append(loss_val.item())
                    aug_lagrangians_val.append([iter, loss_val + not_nlls[-1]])

            # compute delta for lambda
            if iter >= 2 * self.stop_crit_win \
                    and iter % (2 * self.stop_crit_win) == 0:
                t0 = aug_lagrangians_val[-3][1]
                t_half = aug_lagrangians_val[-2][1]
                t1 = aug_lagrangians_val[-1][1]

                # if the validation loss went up and down,
                # do not update lagrangian and penalty coefficients.
                if not (min(t0, t1) < t_half < max(t0, t1)):
                    delta_lambda = -np.inf
                else:
                    delta_lambda = (t1 - t0) / self.stop_crit_win
            else:
                delta_lambda = -np.inf  # do not update lambda nor mu

            # Does the augmented lagrangian converged?
            if h > self.h_threshold:
                # if we have found a stationary point of the augmented loss
                if abs(delta_lambda) < self.omega_lambda or delta_lambda > 0:
                    lamb += mu * h.item()

                    # Did the constraint improve sufficiently?
                    hs.append(h.item())
                    if len(hs) >= 2:
                        if hs[-1] > hs[-2] * self.omega_mu:
                            mu *= 10

                    # little hack to make sure the moving average is going down.
                    with torch.no_grad():
                        gap_in_not_nll = 0.5 * mu * h.item() ** 2 + \
                                         lamb * h.item() - not_nlls[-1]
                        aug_lagrangian_ma[iter + 1] += gap_in_not_nll
                        aug_lagrangians_val[-1][1] += gap_in_not_nll

                    if self.optimizer == "rmsprop":
                        optimizer = torch.optim.RMSprop(self.model.parameters(),
                                                        lr=self.lr)
                    else:
                        optimizer = torch.optim.SGD(self.model.parameters(),
                                                    lr=self.lr)
            else:
                # Final clamping of all edges == 0
                with torch.no_grad():
                    to_keep = (abs(w_adj) > 0).type(torch.Tensor)
                    self.model.adjacency *= to_keep

                return self.model
            
    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=w_prior
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.l1_penalty=lambda1*0.3

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

def compute_constraint(model, w_adj):
    h = torch.trace(torch.matrix_exp(w_adj*w_adj)) - model.input_dim
    return h

class my_NonlinearGaussANM(NonlinearGaussANM):
    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        return my_compute_A_phi(self, norm=self.norm_prod, square=self.square_prod)

def my_compute_A_phi(model, norm="none", square=False):
    """
    compute matrix A consisting of products of NN weights
    """
    weights = model.get_parameters(mode='w')[0]
    prod = torch.eye(model.input_dim)
    if norm != "none":
        prod_norm = torch.eye(model.input_dim)
    for i, w in enumerate(weights):
        if square:
            w = w ** 2
        else:
            # 可以为负
            pass
        if i == 0:
            prod = torch.einsum("tij,ljt,jk->tik", w,
                                model.adjacency.unsqueeze(0), prod)
            if norm != "none":
                tmp = 1. - torch.eye(model.input_dim).unsqueeze(0)
                prod_norm = torch.einsum("tij,ljt,jk->tik",
                                         torch.ones_like(w).detach(), tmp,
                                         prod_norm)
        else:
            prod = torch.einsum("tij,tjk->tik", w, prod)
            if norm != "none":
                prod_norm = torch.einsum("tij,tjk->tik",
                                         torch.ones_like(w).detach(),
                                         prod_norm)

    # sum over density parameter axis
    prod = torch.sum(prod, 1)
    if norm == "paths":
        prod_norm = torch.sum(prod_norm, 1)
        denominator = prod_norm + torch.eye(model.input_dim)  # avoid / 0 on diagonal
        return (prod / denominator).t()
    elif norm == "none":
        return prod.t()
    else:
        raise NotImplementedError
