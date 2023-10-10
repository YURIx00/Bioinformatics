import numpy as np
import logging
import torch
from torch.nn.functional import softmax, cosine_similarity

class Mapper:

    def __init__(
            self,
            S,
            G,
            d=None,
            d_source=None,
            lambda_a=1,
            lambda_d=1,
            lambda_r=1,
            device="cpu",
            adata_map=None,
            random_state=None,
    ):
        """
        定义

        Args:
            S (ndarray): Single nuclei matrix, shape = (number_cell, number_genes).
            G (ndarray): Spatial transcriptions matrix, shape = (number_spots, number_genes).
                Spots can be single cells, or they can contain multiple cells.
            lambda_a (float): Optional. Strength of zbdc loss function. Default is 1.
            lambda_b (float): Optional. Strength of bias. Default is 1.
            device (str or torch.device): Optional. Device is 'cpu'.
            adata_map (scanpy.AnnData): Optional. Mapping initial condition (for resuming previous mappings). Default is None.
            random_state (int): Optional. pass an int to reproduce training. Default is None.
        """
        self.device = device
        self.S = torch.tensor(S, device=self.device, dtype=torch.float32)
        self.G = torch.tensor(G, device=self.device, dtype=torch.float32)

        self.target_density_enabled = d is not None
        if self.target_density_enabled:
            self.d = torch.tensor(d, device=self.device, dtype=torch.float32)

        self.source_density_enabled = d_source is not None
        if self.source_density_enabled:
            self.d_source = torch.tensor(d_source, device=self.device, dtype=torch.float32)

        self.lambda_a = lambda_a
        self.lambda_d = lambda_d
        self.lambda_r = lambda_r
        self.random_state = random_state
        self.activation = torch.nn.Sigmoid()
        self._density_criterion = torch.nn.KLDivLoss(reduction="sum")

        if adata_map is None:
            if self.random_state:
                np.random.seed(seed=self.random_state)
            self.M = np.random.normal(0, 1, (S.shape[0], G.shape[0]))
        else:
            raise NotImplemented

        self.M = torch.tensor(
            self.M, device=device, requires_grad=True, dtype=torch.float32
        )

    def _loss_fn(self):
        """
        Evaluates the loss function.

        Args:
            verbose (bool): Optional. Whether to print the loss results. If True, the loss for each individual term is printed as:
                G term, Bias term. Default is True.

        Returns:
            Tuple of 3 Floats: Total loss, G term, Bias term
        """
        M_probs = softmax(self.M, dim=1)

        if self.target_density_enabled and self.source_density_enabled:
            d_pred = torch.log(
                self.d_source @ M_probs
            )  # KL wants the log in first argument
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)

        elif self.target_density_enabled and not self.source_density_enabled:
            d_pred = torch.log(
                M_probs.sum(axis=0) / self.M.shape[0]
            )  # KL wants the log in first argument
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)
        else:
            density_term = None

        G_pred = torch.matmul(M_probs.t(), self.S)
        gv_term = self.lambda_a * cosine_similarity(G_pred, self.G, dim=1).mean()
        vg_term = self.lambda_a * cosine_similarity(G_pred, self.G, dim=0).mean()
        res_term = self.lambda_r * (torch.log(M_probs) * M_probs).sum()

        total_loss = -gv_term - vg_term + density_term - res_term

        # total_loss = -vg_term - gv_term + density_term + bias_term
        return total_loss, gv_term, vg_term, density_term, res_term

    def train(self, num_epochs, learning_rate=0.1, print_each=50):
        """
        Run the optimizer and returns the mapping outcome.

        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.

        Returns:
            output (ndarray): The optimized mapping matrix M (ndarray), with shape (number_cells, number_spots).
            training_history (dict): loss for each epoch
        """
        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        keys = ["total loss", "gv_term", "vg_term", "density_term" "res_term"]
        values = [[] for i in range(len(keys))]
        training_history = {key: value for key, value in zip(keys, values)}
        for t in range(num_epochs):
            run_loss = self._loss_fn()

            if print_each is None or t % print_each != 0:
                print(f"Epoch {t}, loss = {run_loss[0].item()}, gv_term = {run_loss[1].item()}, vg_term = {run_loss[2].item()},"
                      f" density term = {run_loss[3]}, res term = {run_loss[4].item()  if run_loss[4] is not None else None}")
            loss = run_loss[0]

            for i in range(len(keys)):
                training_history[keys[i]].append(str(run_loss[i]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # take final softmax w/o computing gradients
        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            return output, training_history
