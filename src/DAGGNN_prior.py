from castle.algorithms.gradient.dag_gnn.torch.dag_gnn import *

class myDAG_GNN(DAG_GNN):
    def learn(self, data, columns=None, **kwargs):

        set_seed(self.seed)

        if data.ndim == 2:
            data = data[:,:,None]
        self.n_samples, self.n_nodes, self.input_dim = data.shape

        if self.latent_dim is None:
            self.latent_dim = self.input_dim
        train_loader = func.get_dataloader(data, batch_size=self.batch_size, device=self.device)

        # =====initialize encoder and decoder=====
        adj_A = torch.zeros((self.n_nodes, self.n_nodes), requires_grad=True, device=self.device)
        self.encoder = Encoder(input_dim=self.input_dim,
                               hidden_dim=self.encoder_hidden,
                               output_dim=self.latent_dim,
                               adj_A=adj_A,
                               device=self.device,
                               encoder_type=self.encoder_type.lower()
                               )
        self.decoder = Decoder(input_dim=self.latent_dim,
                               hidden_dim=self.decoder_hidden,
                               output_dim=self.input_dim,
                               device=self.device,
                               decoder_type=self.decoder_type.lower()
                               )
        # =====initialize optimizer=====
        if self.optimizer.lower() == 'adam':
            optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                    {'params': self.decoder.parameters()}],
                                   lr=self.lr)
        elif self.optimizer.lower() == 'sgd':
            optimizer = optim.SGD([{'params': self.encoder.parameters()},
                                   {'params': self.decoder.parameters()}],
                                  lr=self.lr)
        else:
            raise
        self.scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay, gamma=self.gamma)

        ################################
        # main training
        ################################
        c_a = self.init_c_a
        lambda_a = self.init_lambda_a
        h_a_new = torch.tensor(1.0)
        h_a_old = np.inf
        elbo_loss = np.inf
        best_elbo_loss = np.inf
        origin_a = adj_A
        epoch = 0
        for step_k in range(self.k_max_iter):
            while c_a < self.c_a_thresh:
                for epoch in range(self.epochs):
                    elbo_loss, origin_a = self._train(train_loader=train_loader,
                                                      optimizer=optimizer,
                                                      lambda_a=lambda_a,
                                                      c_a=c_a)
                    if elbo_loss < best_elbo_loss:
                        best_elbo_loss = elbo_loss
                if elbo_loss > 2 * best_elbo_loss:
                    break
                # update parameters
                a_new = origin_a.detach().clone()
                h_a_new = func._h_A(a_new, self.n_nodes)
                if h_a_new.item() > self.multiply_h * h_a_old:
                    c_a *= self.eta  # eta
                else:
                    break
            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_a_old = h_a_new.item()
            logging.info(f"Iter: {step_k}, epoch: {epoch}, h_new: {h_a_old}")
            lambda_a += c_a * h_a_new.item()
            if h_a_old <= self.h_tolerance:
                break

        origin_a = origin_a.detach().cpu().numpy()
        origin_a[np.abs(origin_a) < self.graph_threshold] = 0
        origin_a[np.abs(origin_a) >= self.graph_threshold] = 1

        self.causal_matrix = Tensor(origin_a, index=columns, columns=columns)
    
    def _train(self, train_loader, optimizer, lambda_a, c_a):

        self.encoder.train()
        self.decoder.train()

        # update optimizer
        optimizer, lr = func.update_optimizer(optimizer, self.lr, c_a)

        nll_train = []
        kl_train = []
        origin_a = None
        for batch_idx, (data, relations) in enumerate(train_loader):
            x = Variable(data).double()

            optimizer.zero_grad()

            logits, origin_a = self.encoder(x)
            z_gap = self.encoder.z
            z_positive = self.encoder.z_positive
            wa = self.encoder.wa

            x_pred = self.decoder(logits, adj_A=origin_a, wa=wa)    # X_hat

            # reconstruction accuracy loss
            loss_nll = func.nll_gaussian(x_pred, x)

            # KL loss
            loss_kl = func.kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_a = origin_a  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = self.tau_a * torch.sum(torch.abs(one_adj_a))

            # other loss term
            if self.use_a_connect_loss:
                connect_gap = func.a_connect_loss(one_adj_a, self.graph_threshold, z_gap)
                loss += lambda_a * connect_gap + 0.5 * c_a * connect_gap * connect_gap

            if self.use_a_positiver_loss:
                positive_gap = func.a_positive_loss(one_adj_a, z_positive)
                loss += .1 * (lambda_a * positive_gap
                              + 0.5 * c_a * positive_gap * positive_gap)
            # compute h(A)
            h_A = func._h_A(origin_a, self.n_nodes)
            loss += (lambda_a * h_A
                     + 0.5 * c_a * h_A * h_A
                     + 100. * torch.trace(origin_a * origin_a)
                     + sparse_loss
                     + _prior(origin_a,self.w_prior,self.prob_prior))  # +  0.01 * torch.sum(variance * variance)
            if np.isnan(loss.detach().cpu().numpy()):
                raise ValueError(f"The loss value is Nan, "
                                 f"suggest to set optimizer='adam' to solve it. "
                                 f"If you already set, please check your code whether has other problems.")
            loss.backward()
            optimizer.step()
            self.scheduler.step()

            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        return (np.mean(np.mean(kl_train) + np.mean(nll_train)), origin_a)
    
    def load_prior(self,w_prior=None, prob_prior=0):
        self.w_prior=w_prior
        self.prob_prior=prob_prior

    def load_l1_penalty_parameter(self,lambda1=0):
        self.tau_a=lambda1 * 0.01

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
    
    prior*=0.1
    return prior