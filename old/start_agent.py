from sweep_run import train
import wandb

if __name__ == '__main__':
    sweep_id = 'urban-ai/glosa_sweep/uxr9wcor'
    wandb.agent(sweep_id, function=train)
