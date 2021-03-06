import torch
from torch import nn, optim
from os.path import isdir, join
from os import mkdir


class DBCheckpoint:
    def __init__(self, workspace: str, resume: str):
        workspace = join(workspace, 'checkpoint')
        if not isdir(workspace):
            mkdir(workspace)
        workspace = join(workspace, 'detector')
        if not isdir(workspace):
            mkdir(workspace)
        self.workspace: str = workspace
        self.resume: str = resume

    def save(self, model: nn.Module, optimizer: optim.Optimizer,scheduler, epoch: int):
        last_path: str = join(self.workspace, 'last.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }, last_path)
        save_path: str = join(self.workspace, 'checkpoint{}.pth'.format(epoch))
        torch.save({
            'model': model.state_dict()
        }, save_path)

    def load(self):
        if (self.resume is None) or len(self.resume) == 0:
            return None
        state_dict: dict = torch.load(self.resume)
        return state_dict