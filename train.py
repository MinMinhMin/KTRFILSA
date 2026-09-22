import pandas as pd
import numpy as np
import torch
import os
import glob

from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


def _make_training_logger(log_path, data_name, accelerator):
    """Create a flushed stdout/file logger for diagnosing distributed stalls."""
    try:
        log_every_batch = int(os.environ.get("KTRFILSA_LOG_EVERY_BATCH", "0"))
    except ValueError:
        log_every_batch = 0

    if log_every_batch <= 0:
        return lambda event, **fields: None, log_every_batch

    rank = getattr(
        accelerator,
        "process_index",
        getattr(accelerator, "local_process_index", 0),
    )
    world_size = getattr(accelerator, "num_processes", 1)
    debug_dir = os.path.join(log_path, data_name)
    os.makedirs(debug_dir, exist_ok=True)
    debug_file = open(
        os.path.join(debug_dir, f"debug_rank_{rank}.log"),
        "a",
        encoding="utf-8",
        buffering=1,
    )

    def log(event, **fields):
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        details = " ".join(f"{key}={value}" for key, value in fields.items())
        line = (
            f"[train-debug] time={timestamp} rank={rank}/{world_size} "
            f"event={event} {details}"
        ).rstrip()
        print(line, flush=True)
        debug_file.write(line + "\n")
        debug_file.flush()

    return log, log_every_batch


def export_model_state_dict(model, accelerator=None):
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    elif hasattr(model, "module"):
        model = model.module
    state_dict = model.state_dict()
    return {
        key: value.detach().cpu()
        for key, value in state_dict.items()
        if "cluster_friendly_module" not in key
    }


def gather_metric_tensors(accelerator, predictions, truths):
    """Gather variable-length validation/test predictions across DDP workers.

    Filtering padding tokens makes each rank produce a different number of
    predictions. ``Accelerator.gather_for_metrics`` does not pad arbitrary
    tensors before its underlying ``all_gather`` in all supported versions,
    so pad both tensors explicitly and remove the sentinel rows afterwards.
    """
    if getattr(accelerator, "num_processes", 1) == 1:
        return predictions, truths

    if hasattr(accelerator, "pad_across_processes"):
        predictions = accelerator.pad_across_processes(
            predictions, dim=0, pad_index=0.0
        )
        truths = accelerator.pad_across_processes(truths, dim=0, pad_index=-1.0)
        predictions, truths = accelerator.gather((predictions, truths))
        valid = truths > -1
        return predictions[valid], truths[valid]

    # Older Accelerate versions may not expose pad_across_processes. Keep the
    # existing behavior as a compatibility fallback for those installations.
    if hasattr(accelerator, "gather_for_metrics"):
        return accelerator.gather_for_metrics((predictions, truths))
    return accelerator.gather(predictions), accelerator.gather(truths)


def model_train(
    fold,
    model,
    accelerator,
    opt,
    train_loader,
    valid_loader,
    test_loader,
    config,
    n_gpu,
    early_stop=True,
):
    train_losses = []
    avg_train_losses = []
    best_valid_auc = -np.inf
    best_epoch = 0

    logs_df = pd.DataFrame()
    num_epochs = config["train_config"]["num_epochs"]
    model_name = config["model_name"]
    data_name = config["data_name"]
    train_config = config["train_config"]
    log_path = train_config["log_path"]

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time

    debug_log, log_every_batch = _make_training_logger(
        log_path, data_name, accelerator
    )
    debug_log(
        "train_start",
        fold=fold,
        epochs=num_epochs,
        train_batches=len(train_loader),
        valid_batches=len(valid_loader),
        test_batches=len(test_loader),
        device=str(accelerator.device),
    )

    token_cnts = 0
    label_sums = 0
    for i in range(1, num_epochs + 1):
        progress = tqdm(
            total=len(train_loader),
            disable=not accelerator.is_main_process,
            desc=f"train epoch {i}",
        )
        train_iterator = iter(train_loader)
        for batch_idx in range(len(train_loader)):
            should_log_batch = (
                log_every_batch > 0
                and (
                    batch_idx % log_every_batch == 0
                    or batch_idx == len(train_loader) - 1
                )
            )
            if should_log_batch:
                debug_log(
                    "batch_next_start",
                    epoch=i,
                    batch=batch_idx + 1,
                    total_batches=len(train_loader),
                )
            batch = next(train_iterator)
            if should_log_batch:
                debug_log("batch_ready", epoch=i, batch=batch_idx + 1)

            opt.zero_grad()

            model.train()
            if should_log_batch:
                debug_log("forward_start", epoch=i, batch=batch_idx + 1)
            out_dict = model(batch)
            if should_log_batch:
                debug_log("forward_done", epoch=i, batch=batch_idx + 1)

            base_model = accelerator.unwrap_model(model)
            if should_log_batch:
                debug_log("loss_start", epoch=i, batch=batch_idx + 1)
            loss, token_cnt, label_sum = base_model.loss(batch, out_dict)
            if should_log_batch:
                debug_log(
                    "loss_done",
                    epoch=i,
                    batch=batch_idx + 1,
                    loss=f"{loss.item():.6f}",
                )

            if should_log_batch:
                debug_log("backward_start", epoch=i, batch=batch_idx + 1)
            accelerator.backward(loss)
            if should_log_batch:
                debug_log("backward_done", epoch=i, batch=batch_idx + 1)

            token_cnts += token_cnt
            label_sums += label_sum

            if train_config["max_grad_norm"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_config["max_grad_norm"]
                )

            if should_log_batch:
                debug_log("optimizer_start", epoch=i, batch=batch_idx + 1)
            opt.step()
            train_losses.append(loss.item())
            progress.update(1)
            if should_log_batch:
                debug_log("batch_done", epoch=i, batch=batch_idx + 1)

        progress.close()
        debug_log("train_epoch_done", epoch=i)

        print("token_cnts", token_cnts, "label_sums", label_sums, flush=True)

        total_preds = []
        total_trues = []

        with torch.no_grad():
            debug_log("valid_start", epoch=i)
            for batch in valid_loader:
                model.eval()

                out_dict = model(batch)
                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                pred = pred[mask]
                true = true[mask]

                total_preds.append(pred)
                total_trues.append(true)

            debug_log("valid_batches_done", epoch=i)

            debug_log("valid_gather_start", epoch=i)
            total_preds, total_trues = gather_metric_tensors(
                accelerator, torch.cat(total_preds), torch.cat(total_trues)
            )
            debug_log("valid_gather_done", epoch=i)
            total_preds = total_preds.squeeze(-1).detach().cpu().numpy()
            total_trues = total_trues.squeeze(-1).detach().cpu().numpy()

        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        valid_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        checkpoint_root = config.get("checkpoint_dir", "saved_model")
        path = os.path.join(checkpoint_root, model_name, data_name)
        os.makedirs(path, exist_ok=True)

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = i
            if accelerator.is_main_process:
                path = os.path.join(checkpoint_root, model_name, data_name, "params_*")
                for _path in glob.glob(path):
                    os.remove(_path)
                torch.save(
                    {"epoch": i, "model_state_dict": export_model_state_dict(model, accelerator)},
                    os.path.join(checkpoint_root, model_name, data_name, "params_{}".format(str(best_epoch))),
                )
        debug_log("epoch_barrier_start", epoch=i)
        accelerator.wait_for_everyone()
        debug_log("epoch_barrier_done", epoch=i)
        if i - best_epoch > 10:
            break

        # clear lists to track next epochs
        train_losses = []
        valid_losses = []

        total_preds, total_trues = [], []

        # evaluation on test dataset
        with torch.no_grad():
            debug_log("test_start", epoch=i)
            for batch in test_loader:

                model.eval()

                out_dict = model(batch)

                pred = out_dict["pred"].flatten()
                true = out_dict["true"].flatten()
                mask = true > -1
                pred = pred[mask]
                true = true[mask]

                total_preds.append(pred)
                total_trues.append(true)

            debug_log("test_batches_done", epoch=i)

            debug_log("test_gather_start", epoch=i)
            total_preds, total_trues = gather_metric_tensors(
                accelerator, torch.cat(total_preds), torch.cat(total_trues)
            )
            debug_log("test_gather_done", epoch=i)
            total_preds = total_preds.squeeze(-1).detach().cpu().numpy()
            total_trues = total_trues.squeeze(-1).detach().cpu().numpy()

        test_auc = roc_auc_score(y_true=total_trues, y_score=total_preds)

        if accelerator.is_main_process:
            print(
                "Fold {}:\t Epoch {}\t\tTRAIN LOSS: {:.5f}\tVALID AUC: {:.5f}\tTEST AUC: {:.5f}".format(
                    fold, i, train_loss, valid_auc, test_auc
                ),
                flush=True,
            )
    accelerator.wait_for_everyone()
    checkpoint = torch.load(
        os.path.join(checkpoint_root, model_name, data_name, "params_{}".format(str(best_epoch)))
    )

    accelerator.unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )

    total_preds, total_trues = [], []
    total_q_embeds, total_qr_embeds = [], []
    # evaluation on test dataset
    with torch.no_grad():
        for batch in test_loader:

            model.eval()

            out_dict = model(batch)

            pred = out_dict["pred"].flatten()
            true = out_dict["true"].flatten()
            mask = true > -1
            pred = pred[mask]
            true = true[mask]
            total_preds.append(pred)
            total_trues.append(true)

        total_preds, total_trues = gather_metric_tensors(
            accelerator, torch.cat(total_preds), torch.cat(total_trues)
        )
        total_preds = total_preds.squeeze(-1).detach().cpu().numpy()
        total_trues = total_trues.squeeze(-1).detach().cpu().numpy()

    auc = roc_auc_score(y_true=total_trues, y_score=total_preds)
    acc = accuracy_score(y_true=total_trues >= 0.5, y_pred=total_preds >= 0.5)
    rmse = np.sqrt(mean_squared_error(y_true=total_trues, y_pred=total_preds))

    if accelerator.is_main_process:
        print(
            "Best Model\tTEST AUC: {:.5f}\tTEST ACC: {:5f}\tTEST RMSE: {:5f}".format(
                auc, acc, rmse
            ),
            flush=True,
        )

    logs_df = pd.concat(
        [
            logs_df,
            pd.DataFrame(
                {
                    "EarlyStopEpoch": [best_epoch],
                    "auc": [auc],
                    "acc": [acc],
                    "rmse": [rmse],
                }
            ),
        ],
        ignore_index=True,
    )

    if accelerator.is_main_process:
        log_out_path = os.path.join(log_path, data_name)
        os.makedirs(log_out_path, exist_ok=True)
        logs_df.to_csv(
            os.path.join(log_out_path, "{}_{}.csv".format(model_name, now)), index=False
        )

    return auc, acc, rmse
