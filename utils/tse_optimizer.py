import torch


class TSEOptimizer:
    """
    Twin-Stars Entwined style optimizer built on top of a base optimizer.

    It keeps a shadow endpoint for each parameter and uses the current
    distance between model parameters and shadow parameters to set the
    perturbation radius adaptively (SAM-like first/second step).
    """

    def __init__(self, params, base_optimizer, rho_min=1e-4, rho_max=0.5, distance_lambda=1e-4, shadow_momentum=0.05):
        self.base_optimizer = base_optimizer
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.distance_lambda = distance_lambda
        self.shadow_momentum = shadow_momentum
        self._last_rho = rho_min

        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.base_optimizer.state[p]
                    state['shadow'] = p.detach().clone()

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def distance_regularization(self):
        reg = None
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                shadow = self.base_optimizer.state[p]['shadow']
                term = torch.sum((p - shadow) ** 2)
                reg = term if reg is None else reg + term
        if reg is None:
            return torch.tensor(0.0)
        return self.distance_lambda * reg

    def _adaptive_rho(self):
        total_sq = None
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                shadow = self.base_optimizer.state[p]['shadow']
                sq = torch.sum((p - shadow) ** 2)
                total_sq = sq if total_sq is None else total_sq + sq

        if total_sq is None:
            self._last_rho = self.rho_min
            return self._last_rho

        distance = torch.sqrt(total_sq).item()
        rho = max(self.rho_min, min(self.rho_max, 0.5 * distance))
        self._last_rho = rho
        return rho

    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        rho = self._adaptive_rho()
        scale = rho / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.base_optimizer.state[p]['e_w'] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.base_optimizer.state[p]['e_w'])

        self.base_optimizer.step()

        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.base_optimizer.state[p]
                shadow = state['shadow']
                shadow.add_(self.shadow_momentum * (p.detach() - shadow))

        if zero_grad:
            self.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
