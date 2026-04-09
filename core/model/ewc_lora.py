# EWC-LoRA (ICLR 2026). Adapted from https://github.com/yaoyz96/low-rank-cl

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .finetune import Finetune


class FisherComputer:
    def __init__(self, network, dataloader, label_offset, criterion, device=torch.device("cpu")):
        self.model = network.to(device)
        self.dataloader = dataloader
        self.label_offset = label_offset
        self.criterion = criterion
        self.device = device
        self.fisher_W = []
        self._init_fisher_storage()

    def compute(self, max_batches=None):
        self.model.eval()
        num_samples = 0

        for i, batch in enumerate(tqdm(self.dataloader, desc="Computing Fisher")):
            if max_batches is not None and i >= max_batches:
                break
            inputs = batch["image"].to(self.device)
            targets = batch["label"].to(self.device)
            self.model.zero_grad()
            logits = self.model.forward(inputs, use_new=True, register_hook=True)["logits"]
            targets = targets - self.label_offset
            loss = self.criterion(logits, targets)
            loss.backward()

            batch_size = inputs.size(0)
            num_samples += batch_size

            idx = 0
            for module in self.model.modules():
                if hasattr(module, "delta_w_k_new_grad"):
                    grad_k = module.delta_w_k_new_grad
                    if grad_k is not None:
                        self.fisher_W[idx] += (grad_k.detach() ** 2) * batch_size
                    idx += 1
                if hasattr(module, "delta_w_v_new_grad"):
                    grad_v = module.delta_w_v_new_grad
                    if grad_v is not None:
                        self.fisher_W[idx] += (grad_v.detach() ** 2) * batch_size
                    idx += 1

        self.fisher_W = [fw / num_samples for fw in self.fisher_W]
        return self.fisher_W

    def _init_fisher_storage(self):
        for module in self.model.modules():
            if hasattr(module, "lora_new_B_k") and hasattr(module, "lora_new_A_k"):
                delta_w_k_new = module.lora_new_B_k.weight @ module.lora_new_A_k.weight
                self.fisher_W.append(torch.zeros_like(delta_w_k_new))
            if hasattr(module, "lora_new_B_v") and hasattr(module, "lora_new_A_v"):
                delta_w_v_new = module.lora_new_B_v.weight @ module.lora_new_A_v.weight
                self.fisher_W.append(torch.zeros_like(delta_w_v_new))


class EWCLoRA(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__(backbone, feat_dim, num_class, **kwargs)
        self._network = backbone
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.init_cls_num = kwargs.get("init_cls_num", self.inc_cls_num)
        self.total_sessions = kwargs["total_sessions"]
        self.device = kwargs["device"]

        self.gamma = kwargs.get("gamma", 1.0)
        self.ewc_weight = kwargs.get("lambda_ewc", kwargs.get("ewc_lambda", 1e7))
        self.encoder_lr = kwargs.get("encoder_lr", 5e-4)
        self.fc_lr = kwargs.get("fc_lr", 5e-3)
        self.weight_decay = kwargs.get("weight_decay", 0.0)

        self._total_classes = 0
        self._known_classes = 0
        self._cur_task = -1

        self.omega_W = []
        self.count_updates = 0
        self._fisher_label_offset = 0

    def get_parameters(self, config):
        _ = config
        enc_params = []
        cls_params = []
        for name, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            if "classifier_pool" in name:
                cls_params.append(p)
            else:
                enc_params.append(p)
        return [
            {"params": enc_params, "lr": self.encoder_lr, "weight_decay": self.weight_decay},
            {"params": cls_params, "lr": self.fc_lr, "weight_decay": self.weight_decay},
        ]

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        self._known_classes = self._total_classes
        self._cur_task += 1
        n_new = self.init_cls_num if task_idx == 0 else self.inc_cls_num
        self._total_classes = self._known_classes + n_new
        self._fisher_label_offset = self._known_classes
        self._network.update_fc(self._total_classes)
        self._network.to(self.device)

        for _, param in self._network.named_parameters():
            param.requires_grad_(False)

        suffix = f".{self._cur_task}"
        keys = (
            f"classifier_pool{suffix}",
            "lora_new_A_k",
            "lora_new_B_k",
            "lora_new_A_v",
            "lora_new_B_v",
        )
        for name, param in self._network.named_parameters():
            if any(k in name for k in keys):
                param.requires_grad_(True)

    def observe(self, data):
        x, y = data["image"], data["label"]
        x = x.to(self.device)
        y = y.to(self.device)

        mask = ((y >= self._known_classes) & (y < self._total_classes)).nonzero().view(-1)
        if mask.numel() == 0:
            p = next(p for p in self._network.parameters() if p.requires_grad)
            z = (p * 0).sum()
            return torch.zeros(1, device=self.device, dtype=torch.long), 0.0, z

        x = torch.index_select(x, 0, mask)
        y = torch.index_select(y, 0, mask) - self._known_classes

        logits = self._network(x, use_new=True)["logits"]
        loss = F.cross_entropy(logits, y)

        if self.count_updates != 0:
            ewc_loss = 0.0
            new_a_params = filter(lambda p: getattr(p, "_is_new_a", False), self._network.parameters())
            new_b_params = filter(lambda p, getattr(p, "_is_new_b", False), self._network.parameters())
            for idx, (p_a, p_b) in enumerate(zip(new_a_params, new_b_params)):
                delta_W = p_b @ p_a
                ewc_term = self.omega_W[idx].type(torch.float32).to(self.device) * (delta_W**2)
                ewc_loss += torch.sum(ewc_term)
            loss = loss + (self.ewc_weight / 2.0) * ewc_loss

        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()
        return pred, acc, loss

    def inference(self, data):
        x, y = data["image"], data["label"]
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self._network.interface(x, use_new=True)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == y).float().mean().item()
        return pred, acc

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        from torch.utils.data import DataLoader

        _ = buffer, test_loaders
        loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=getattr(train_loader, "num_workers", 0),
            drop_last=getattr(train_loader, "drop_last", False),
            pin_memory=getattr(train_loader, "pin_memory", False),
        )

        self.count_updates += 1
        fisher = FisherComputer(
            self._network,
            loader,
            self._fisher_label_offset,
            F.cross_entropy,
            self.device,
        )
        fisher_W = fisher.compute(max_batches=None)

        omega_W_bk = self.omega_W[:]
        self.omega_W = []

        new_a_params = filter(lambda p: getattr(p, "_is_new_a", False), self._network.parameters())
        new_b_params = filter(lambda p: getattr(p, "_is_new_b", False), self._network.parameters())
        for idx, (_p_a, _p_b) in enumerate(zip(new_a_params, new_b_params)):
            if len(omega_W_bk) != 0:
                self.omega_W.append(self.gamma * omega_W_bk[idx] + fisher_W[idx])
            else:
                self.omega_W.append(fisher_W[idx])

        self._network.accumulate_and_reset_lora()


def run_reference_single_task_finetunes_ewclora(
    config,
    device,
    task_num,
    init_cls_num,
    inc_cls_num,
    train_loader_bundle,
    test_loader_bundle,
    init_epoch,
    inc_epoch,
):
    """
    Isolated fine-tuning per task for A^{ref}_{i,i} in Zheng et al. (2026) Eq. (6).

    For each task, builds a fresh SiNet_vit_ewclora (total_sessions=1), trains only on that task's
    data with the same LoRA setup as EWC-LoRA (no EWC penalty), same epochs / Adam / cosine as the
    main config, then evaluates on that task's test set. Returns reference accuracies in %% (0--100),
    same scale as LibContinual's acc_table diagonal.
    """
    import copy
    import gc

    import torch
    from torch.optim import Adam

    from core.scheduler import CosineSchedule
    from core.utils.utils import init_seed

    from .backbone.vit_ewclora import Attention_LoRA, SiNet_vit_ewclora

    c_kw = config["classifier"]["kwargs"]
    encoder_lr = float(c_kw.get("encoder_lr", 5e-4))
    fc_lr = float(c_kw.get("fc_lr", 5e-3))
    wd = float(c_kw.get("weight_decay", 0.0))
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 0)
    pin_memory = config.get("pin_memory", False)

    ref_accs = []

    for task_idx in range(task_num):
        init_seed(
            int(config["seed"]) + 10007 + task_idx * 17,
            bool(config.get("deterministic", False)),
        )

        n_cls = init_cls_num if task_idx == 0 else inc_cls_num
        label_offset = 0 if task_idx == 0 else init_cls_num + (task_idx - 1) * inc_cls_num
        epochs = int(init_epoch if task_idx == 0 else inc_epoch)

        bb_kw = copy.deepcopy(config["backbone"]["kwargs"])
        bb_kw["total_sessions"] = 1
        bb_kw["init_cls"] = n_cls

        net = SiNet_vit_ewclora(**bb_kw)
        net.to(device)
        for module in net.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        net._cur_task = -1
        net.update_fc(n_cls)

        for _, param in net.named_parameters():
            param.requires_grad_(False)
        suffix = ".0"
        unfrozen = (
            f"classifier_pool{suffix}",
            "lora_new_A_k",
            "lora_new_B_k",
            "lora_new_A_v",
            "lora_new_B_v",
        )
        for name, param in net.named_parameters():
            if any(k in name for k in unfrozen):
                param.requires_grad_(True)

        enc_params = []
        cls_params = []
        for name, p in net.named_parameters():
            if not p.requires_grad:
                continue
            if "classifier_pool" in name:
                cls_params.append(p)
            else:
                enc_params.append(p)
        optimizer = Adam(
            [
                {"params": enc_params, "lr": encoder_lr, "weight_decay": wd},
                {"params": cls_params, "lr": fc_lr, "weight_decay": wd},
            ],
            betas=(0.9, 0.999),
        )
        scheduler = CosineSchedule(optimizer, K=max(epochs, 1))

        train_loader = train_loader_bundle.get_loader(task_idx)
        test_loaders = test_loader_bundle.get_loader(task_idx)
        test_loader = test_loaders[task_idx]

        net.train()
        for _epoch in range(epochs):
            for batch in tqdm(
                train_loader,
                desc=f"Ref task {task_idx} ep {_epoch + 1}/{epochs}",
                leave=False,
            ):
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                y_loc = y - label_offset
                mask = (y_loc >= 0) & (y_loc < n_cls)
                if mask.sum().item() == 0:
                    continue
                x = x[mask]
                y_loc = y_loc[mask]

                logits = net(x, use_new=True)["logits"]
                loss = F.cross_entropy(logits, y_loc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                y_loc = y - label_offset
                mask = (y_loc >= 0) & (y_loc < n_cls)
                if mask.sum().item() == 0:
                    continue
                x = x[mask]
                y_loc = y_loc[mask]
                logits = net.interface(x, use_new=True)
                pred = logits.argmax(dim=1)
                correct += (pred == y_loc).sum().item()
                total += y_loc.numel()

        acc_pct = 100.0 * correct / max(total, 1)
        ref_accs.append(acc_pct)

        del net, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ref_accs
