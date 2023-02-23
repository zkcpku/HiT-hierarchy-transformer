import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, peak):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        self.original_peak = np.power((n_warmup_steps * d_model), -0.5)
        self.count = peak / self.original_peak

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale() * self.count

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    # peak = 1/(sqrt(warmup_step*d_model)) = 4.94e-4
    # 4000 -> 1e-4
    # 1ep=3221
    # 10ep -> 32221 ->  np.pow(32221*1024,-0.5) *


if __name__ == '__main__':
    original_peak = np.power((4000 * 1024), -0.5)
    count = 1e-4 / original_peak
    x = 5336
    print(np.power(x * 1024, -0.5) * count)
    # 感觉还可以
