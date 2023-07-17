import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.cli import LightningCLI

class MyDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, input_size, batch_size, data_info):
        super(PowerLineDataModule).__init__()
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.data_info = data_info

        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass


    def setup(self, stage: str = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Lists that store the Dataset classes for training, validation, and testing. They will concatenated into a single training, validation, or testing dataset
        training_datasets = []
        valid_datasets = []
        test_datasets = []

        if stage == "fit" or stage is None:
	        print("Stage is fit")
	        self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            print("Stage is test")
            
    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet. 
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count()-4)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=os.cpu_count()-4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=os.cpu_count()-4)