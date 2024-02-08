

from lightning.pytorch.demos.boring_classes import DemoModel

class Model1(DemoModel):

    def __init__(self, learning_rate, test):
        """
        learning_rate : learning rate in check
        """
        super()
        pass

    def configure_optimizers(self):
        print("⚡", "using Model1", "⚡")
        return super().configure_optimizers()


class Model2(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model2", "⚡")
        return super().configure_optimizers()
