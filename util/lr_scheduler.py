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
    def __init__(self, decay_rate=0.1, lr_milestones=[20, 40], warm_up_epoch=5) -> None:
        self.deacy_rate = decay_rate
        self.lr_milestones = lr_milestones
        self.warm_up_epoch = warm_up_epoch

    def __call__(self, epoch):
        if epoch < self.warm_up_epoch:
            return (epoch+1)/self.warm_up_epoch
        else:
            return self.deacy_rate ** len([m for m in self.lr_milestones if m <= epoch])


class CosineWarmupLR:
    def __init__(self, max_epoch=100, warm_up_epoch=5) -> None:
        self.max_epoch = max_epoch
        self.warm_up_epoch = warm_up_epoch

    def __call__(self, epoch):
        if epoch < self.warm_up_epoch:
            return (epoch+1)/self.warm_up_epoch
        else:
            return 0.5 * (math.cos((epoch - self.warm_up_epoch) / (self.max_epoch - self.warm_up_epoch) * math.pi) + 1)