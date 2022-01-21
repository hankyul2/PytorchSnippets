"""
How to use wandb in pytorch & distributed setup?

1. init & log in root processor
2. set project name
3. set run_name (called name) and link this variable to your local log folder

"""

import wandb


def run_experiment(project_name, run_name, config):
    
    wandb.init(project=project_name, name=run_name, config=config)

    for i in range(100):
        wandb.log({'train_acc': i*config['lr'], 'val_acc': i*config['lr']})


if __name__ == '__main__':
    project_name = 'pt_wandb'
    config = {'model': 'hello', 'batch_size': 128, 'lr': 0.01}

    run_experiment(project_name, 'exp1', config)