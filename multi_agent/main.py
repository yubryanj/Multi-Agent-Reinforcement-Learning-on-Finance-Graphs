from trainer import Trainer
from common.args import get_args
from common.utils import make_env

if __name__ == "__main__":
    # Retrieve the arguments
    args = get_args()

    # Initialize the environment
    env, args = make_env(args)

    # Initialize the trainer module
    trainer = Trainer(args, env)

    # If the evaulation flag is enabled, evaluate the model
    if args.evaluate:
        returns = trainer.evaluate()
        print(f'Average return is {returns}')
    else:
        # Else train the model
        trainer.train()