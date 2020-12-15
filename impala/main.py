from impala import helpers
from impala.trainer import Trainer


if __name__ == '__main__':
    helpers.logging_setup()
    trainer = Trainer()
    trainer.run()