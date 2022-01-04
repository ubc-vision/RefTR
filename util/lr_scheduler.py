import math
# config["max_num_epochs"] = 100
# warm_up_epochs = 5
# lr_milestones = [20,40]
# # MultiStepLR without warm up
# multistep_lr = lambda epoch: 0.1**len([m for m in lr_milestones if m <= epoch])
# # warm_up_with_multistep_lr
# warm_up_with_multistep_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs else 0.1**len([m for m in lr_milestones if m <= epoch])
# # warm_up_with_step_lr
# gamma = 0.9; stepsize = 1
# warm_up_with_step_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
#     else gamma**( ((epoch - warm_up_epochs) /(config["max_num_epochs"] - warm_up_epochs))//stepsize*stepsize)
# # warm_up_with_cosine_lr
# warm_up_with_cosine_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
#     else 0.5 * ( math.cos((epoch - warm_up_epochs) /(config["max_num_epochs"] - warm_up_epochs) * math.pi) + 1)

# scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

class MultiStepWarmupLR:
    def __init__(self, decay_rate=0.1, lr_milestones=[20000, 40000], warm_up_steps=5000, min_decay_rate=0.01) -> None:
        self.deacy_rate = decay_rate
        self.lr_milestones = lr_milestones
        self.warm_up_steps = warm_up_steps
        self.min_decay_rate = min_decay_rate

    def __call__(self, steps):
        if steps < self.warm_up_steps:
            rate = (steps+1)/self.warm_up_steps
        else:
            rate = self.deacy_rate ** len([m for m in self.lr_milestones if m <= steps])
        # make sure lr is not too small
        if rate <= self.min_decay_rate:
            return self.min_decay_rate
        else:
            return rate

class CosineWarmupLR:
    def __init__(self, max_T=100, warm_up_steps=5, min_decay_rate=0.01) -> None:
        self.max_T = max_T
        self.warm_up_steps = warm_up_steps
        self.min_decay_rate = min_decay_rate

    def __call__(self, steps):
        if steps < self.warm_up_steps:
            rate = (steps+1)/self.warm_up_steps
        else:
            rate = 0.5 * (math.cos((steps - self.warm_up_steps) / (self.max_T - self.warm_up_steps) * math.pi) + 1)
        # make sure lr is not too small
        if rate <= self.min_decay_rate:
            return self.min_decay_rate
        else:
            return rate
