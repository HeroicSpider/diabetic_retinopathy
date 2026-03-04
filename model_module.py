import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics import CohenKappa, F1Score
from torchmetrics.classification import BinaryAccuracy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class DRClassifier(pl.LightningModule):
    def __init__(self, model_name='efficientnet_b3', num_classes=5, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.kappa = CohenKappa(task='multiclass', num_classes=num_classes, weights='quadratic')
        self.f1_macro = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.ref_acc = BinaryAccuracy()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.log_temperature = nn.Parameter(torch.zeros(1)) 

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x) / self.temperature

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=x.size(0))
        self.log('val_kappa', self.kappa(preds, y), on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log('val_f1_macro', self.f1_macro(preds, y), on_epoch=True, batch_size=x.size(0))
        
        y_binary = (y >= 2).long()
        preds_binary = (preds >= 2).long()
        self.log('val_referable_acc', self.ref_acc(preds_binary, y_binary), on_epoch=True, batch_size=x.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def calibrate_temperature(self, val_dataloader):
        """
        Optimizes log_temperature on CPU. 
        """
        self.eval()
        self.to('cpu') 
        
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.01, max_iter=50)

        print("Collecting logits for calibration (CPU)...")
        logits_list = []
        labels_list = []
        
        with torch.inference_mode():
            for batch in val_dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
                
                # Concise CPU move (no redundant copies)
                x = x.cpu() if x.device.type != 'cpu' else x
                logits_list.append(self(x).detach())
                
                if y is not None:
                    y = y.cpu() if y.device.type != 'cpu' else y
                    labels_list.append(y.detach())
        
        if not logits_list:
            print("Warning: Validation loader empty. Resetting temperature.")
            self.log_temperature.data.zero_() 
            return {"final_temp": 1.0, "log_temp": 0.0, "nll_improvement": 0.0}
            
        if not labels_list:
             raise ValueError("Calibration failed: Validation loader yields no labels.")

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        before_nll = nll_criterion(logits, labels).item()

        def closure():
            optimizer.zero_grad()
            
            # Clamp BEFORE forward to ensure valid loss
            with torch.no_grad():
                self.log_temperature.clamp_(min=-5.0, max=5.0)

            with torch.enable_grad():
                loss = nll_criterion(logits / self.temperature, labels)
                loss.backward()
            
            return loss
            
        optimizer.step(closure)
        
        after_nll = nll_criterion(logits / self.temperature, labels).item()
        final_temp = self.temperature.item()

        print(f"Calibration Complete. Temp: {final_temp:.4f} | NLL: {before_nll:.4f} -> {after_nll:.4f}")
        
        return {
            "final_temp": final_temp, 
            "log_temp": self.log_temperature.item(),
            "nll_improvement": before_nll - after_nll
        }