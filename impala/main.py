import helpers
from trainer import Trainer


if __name__ == '__main__':
    helpers.logging_setup()
    trainer = Trainer()
    trainer.run()
