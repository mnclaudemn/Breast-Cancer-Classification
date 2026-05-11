import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
# ===================== CONFIG =====================
@dataclass
class TrainingConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4

    mixed_precision: str = "fp16"  # "fp16", "bf16", or None
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    device: str = "cuda"


# ===================== METRIC =====================

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean()


# ===================== CALLBACK BASE =====================

class Callback:
    def on_train_begin(self, trainer): pass
    def on_train_end(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_epoch_end(self, trainer, epoch, logs=None): pass
    def on_step_end(self, trainer, step, logs=None): pass
    def on_validation_end(self, trainer, logs=None): pass


# ===================== CALLBACKS =====================

class EarlyStoppingCallback(Callback):
    def __init__(self, patience=3, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def on_validation_end(self, trainer, logs=None):
        metric = logs["val_acc"] if self.mode == "max" else logs["val_loss"]

        if self.best is None:
            self.best = metric
            return

        improved = (
            metric > self.best + self.min_delta if self.mode == "max"
            else metric < self.best - self.min_delta
        )

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            trainer.should_stop = True


class ModelCheckpointCallback(Callback):
    def __init__(self, path="best_model.pth", mode="max"):
        self.path = path
        self.mode = mode
        self.best = None

    def on_validation_end(self, trainer, logs=None):
        metric = logs["val_acc"] if self.mode == "max" else logs["val_loss"]

        if self.best is None:
            self.best = metric
            torch.save(trainer.model.state_dict(), self.path)
            return

        improved = metric > self.best if self.mode == "max" else metric < self.best

        if improved:
            self.best = metric
            torch.save(trainer.model.state_dict(), self.path)


class LoggingCallback(Callback):
    def on_epoch_end(self, trainer, epoch, logs=None):
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {logs['train_loss']:.4f} | "
            f"Train Acc: {logs['train_acc']:.4f} | "
            f"Val Loss: {logs['val_loss']:.4f} | "
            f"Val Acc: {logs['val_acc']:.4f}"
        )


# ===================== TRAINER =====================

class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig):
        self.cfg = config

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        self.criterion = nn.CrossEntropyLoss()

        # AMP setup
        self.use_amp = config.mixed_precision in ["fp16", "bf16"]
        self.autocast_dtype = (
            torch.float16 if config.mixed_precision == "fp16"
            else torch.bfloat16
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(config.mixed_precision == "fp16")
        )

        self.callbacks = []
        self.should_stop = False

    # ---------- CALLBACK SYSTEM ----------
    def add_callback(self, callback):
        self.callbacks.append(callback)

    def call(self, event, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, event)(self, *args, **kwargs)

    # ---------- TRAIN ONE EPOCH ----------
    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0

        for step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.cuda.amp.autocast(
                enabled=self.use_amp,
                dtype=self.autocast_dtype
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.cfg.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)

                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.cfg.grad_accum_steps
            total_acc += accuracy(outputs, labels).item()

            self.call("on_step_end", step, {"loss": loss.item()})

        return (
            total_loss / len(self.train_loader),
            total_acc / len(self.train_loader)
        )

    # ---------- VALIDATION ----------
    def evaluate(self):
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast(
                    enabled=self.use_amp,
                    dtype=self.autocast_dtype
                ):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                total_acc += accuracy(outputs, labels).item()

        return (
            total_loss / len(self.val_loader),
            total_acc / len(self.val_loader)
        )

    # ---------- TRAIN LOOP ----------
    def train(self):
        self.call("on_train_begin")

        for epoch in range(self.cfg.epochs):
            if self.should_stop:
                break

            self.call("on_epoch_begin", epoch)

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate()

            self.scheduler.step()

            logs = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }

            self.call("on_epoch_end", epoch, logs)
            self.call("on_validation_end", logs)

        self.call("on_train_end")
