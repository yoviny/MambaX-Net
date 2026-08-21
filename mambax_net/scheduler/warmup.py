# original version - https://github.com/ildoonet/pytorch-gradual-warmup-lr
# modified version - https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/util.py

from warmup_scheduler import GradualWarmupScheduler


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """Gradual learning-rate warmup scheduler with a fixed `get_lr` for `multiplier == 1.0`.

    Subclasses `GradualWarmupScheduler` and overrides `get_lr` so that when
    `multiplier` is 1.0 the learning rate ramps linearly from 0 up to the
    base learning rate over `total_epoch` steps (the parent class's
    `get_lr` does not handle this case correctly).
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        """Gradually warm-up(increasing) learning rate in optimizer.
        Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            multiplier (float): target learning rate = base lr * multiplier
            total_epoch (int): target learning rate is reached at total_epoch, gradually
            after_scheduler (torch.optim.lr_scheduler.*): after target_epoch, use this scheduler(eg. ReduceLROnPlateau)

        Raises:
            None

        Example:
            >>> scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10)
            >>> for epoch in range(1, total_epoch+1):
            >>>     scheduler.step()
            >>>     train(...)
            >>>     validate(...)
        """
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler
        )

    def get_lr(self):
        """Get updated learning rate.

        Args:
            None

        Raises:
            None

        Returns:
            list: The new learning rate
        """
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
