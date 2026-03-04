# train.py (Fixed Version)
import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger

from data_module import DRDataModule
from model_module import DRClassifier
from model_module import DRRegressor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--tracking_uri', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default='DR_Diagnosis_Stack')
    
    # Hparams
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='efficientnet_b3')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=4.0)
    parser.add_argument('--beta', type=float, default=10.0)
    
    # ADDED THIS LINE:
    parser.add_argument('--default_root_dir', type=str, default=None, help="Path for logs and checkpoints")
    
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')
    
    # 1. Init Data
    dm = DRDataModule(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        fold_idx=args.fold_idx,
        image_size=args.image_size,
        alpha=args.alpha,
        beta=args.beta,
        use_graham=True 
    )
    
    # 2. Init Model
    model = DRRegressor(               # <--- CHANGED FROM DRClassifier
        model_name=args.model_name,
        lr=args.lr
    )
    
    # 3. Init Logger
    # Use default_root_dir for local logs if tracking_uri isn't set
    mlf_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"),
        save_dir=args.default_root_dir # Store MLflow runs here if local
    )
    
    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_kappa',
        mode='max',
        filename='{epoch}-{val_kappa:.4f}',
        save_top_k=1,
        save_last=True
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=mlf_logger,
        default_root_dir=args.default_root_dir, # PASSED TO TRAINER
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor='val_kappa', mode='max', patience=5, verbose=True),
            LearningRateMonitor(logging_interval='epoch')
        ],
        log_every_n_steps=10
    )
    
    # 5. Train
    trainer.fit(model, datamodule=dm)
'''    
    # 6. Calibrate & Persist
    best_path = checkpoint_callback.best_model_path
    if best_path:
        print(f"Loading best model from {best_path} for calibration...")
        best_model = DRClassifier.load_from_checkpoint(best_path)
        
        dm.setup()
        metrics = best_model.calibrate_temperature(dm.val_dataloader())
        
        run_id = mlf_logger.run_id
        mlf_logger.experiment.log_metric(run_id, "final_temperature", metrics['final_temp'])
        mlf_logger.experiment.log_metric(run_id, "calibration_nll_gain", metrics['nll_improvement'])
        
        calibrated_path = best_path.replace(".ckpt", "_calibrated_bundle.pth")
        bundle = {
            "state_dict": best_model.state_dict(),
            "hparams": dict(best_model.hparams), 
            "calibration_metrics": metrics
        }
        torch.save(bundle, calibrated_path)
        print(f"Calibrated bundle saved to: {calibrated_path}")
        mlf_logger.experiment.log_artifact(run_id, calibrated_path, artifact_path="models")

        '''
if __name__ == '__main__':
    main()
