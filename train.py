import torch
from torch import nn
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from teeth_dataset import TeethSegmentationDataset


class TeethSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, img_dir, ann_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        full_dataset = TeethSegmentationDataset(self.img_dir, self.ann_dir)

        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=int(self.batch_size / 2),
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=int(self.batch_size / 2),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=int(self.batch_size / 2),
            num_workers=self.num_workers,
        )


# Define a double convolution block
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, num_classes=32):
        super(UNet, self).__init__()

        self.encoder1 = double_conv(1, 64)
        self.encoder2 = double_conv(64, 128)
        self.encoder3 = double_conv(128, 256)
        self.encoder4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = double_conv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = double_conv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = double_conv(128, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final(dec1)


# Define the LightningModule
class UNetLightning(pl.LightningModule):
    def __init__(self, num_classes=32):
        super(UNetLightning, self).__init__()
        self.model = UNet()
        self.criterion = nn.CrossEntropyLoss()
        self.iou_metric = MulticlassJaccardIndex(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks.long())
        iou = self.iou_metric(outputs.argmax(dim=1), masks)
        self.log("test_loss", loss)
        self.log("test_iou", iou)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    filename="teeth-seg-{epoch:02d}-{val_loss:.2f}",
)
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=0.001, patience=5, verbose=True, mode="min"
)

# Logger
mlflow_logger = MLFlowLogger(
    experiment_name="teeth_segmentation", tracking_uri="./mlruns"
)

# Trainer
trainer = pl.Trainer(
    max_epochs=10,  # Set to 100 for full training
    accelerator="auto",
    precision=16,
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=mlflow_logger,
)


def main():
    # Initialize data module
    dm = TeethSegmentationDataModule(img_dir="data/d2/img", ann_dir="data/d2/ann")

    model = UNetLightning()

    trainer.fit(model, dm)

    torch.save(model.state_dict(), "unet_model.pth")
    print("Model saved to unet_model.pth")

    trainer.test(model, dm)


if __name__ == "__main__":
    main()
