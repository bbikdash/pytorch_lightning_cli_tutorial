from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

class MyDataModule(LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, input_size, batch_size, data_info):
        super(MyDataModule).__init__()
        self.input_size = input_size
        self.batch_size = batch_size
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
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=4)