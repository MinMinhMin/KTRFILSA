import os
import argparse
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import yaml
from data_loaders import (
    MostRecentQuestionSkillDataset,
    MostEarlyQuestionSkillDataset,
    SimCLRDatasetWrapper,
)
from models.cl4kt import CL4KT
from train import model_train
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from utils.config import ConfigNode as CN
from utils.file_io import PathManager


def main(config, gpu=None, ddp=False):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    accelerator = Accelerator()
    device = accelerator.device
    distributed = accelerator.num_processes > 1
    if ddp and not distributed:
        raise RuntimeError(
            "--ddp requires a torchrun/accelerate distributed launch; "
            "no distributed workers were detected."
        )

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name
    seed = config.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    df_path = os.path.join(os.path.join(dataset_path, data_name), "preprocessed_df.csv")

    train_config = config.train_config
    checkpoint_dir = config.checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_path = os.path.join(checkpoint_dir, model_name)
    os.makedirs(ckpt_path, exist_ok=True)

    ckpt_path = os.path.join(ckpt_path, data_name)
    os.makedirs(ckpt_path, exist_ok=True)

    batch_size = train_config.batch_size
    eval_batch_size = train_config.eval_batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer
    seq_len = train_config.seq_len

    if train_config.sequence_option == "recent":  # the most recent N interactions
        dataset = MostRecentQuestionSkillDataset
    elif train_config.sequence_option == "early":  # the most early N interactions
        dataset = MostEarlyQuestionSkillDataset
    else:
        raise NotImplementedError("sequence option is not valid")

    test_aucs, test_accs, test_rmses = [], [], []

    df = pd.read_csv(df_path, sep="\t")

    print("skill_min", df["skill_id"].min())
    users = df["user_id"].unique()
    df["skill_id"] += 1  # zero for padding
    df["item_id"] += 1  # zero for padding
    num_skills = df["skill_id"].max() + 1
    num_questions = df["item_id"].max() + 1

    official_split = "split" in df.columns and {"train", "valid", "test"}.issubset(set(df["split"].unique()))
    if official_split:
        split_defs = [
            (
                0,
                df.loc[df["split"] == "train", "user_id"].unique(),
                df.loc[df["split"] == "valid", "user_id"].unique(),
                df.loc[df["split"] == "test", "user_id"].unique(),
            )
        ]
        print("Using official train/valid/test split from preprocessed_df.csv")
    else:
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        users = users.copy()
        np.random.shuffle(users)
        split_defs = []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(users)):
            train_users = users[train_ids]
            np.random.shuffle(train_users)
            offset = int(len(train_ids) * 0.9)
            split_defs.append((fold, train_users[:offset], train_users[offset:], users[test_ids]))

    print("MODEL", model_name)
    print(dataset)

    for fold, train_users, valid_users, test_users in split_defs:
        if model_name == "cl4kt":
            model_config = config.cl4kt_config
            model = CL4KT(num_skills, num_questions, seq_len, **model_config)
            mask_prob = model_config.mask_prob
            crop_prob = model_config.crop_prob
            permute_prob = model_config.permute_prob
            replace_prob = model_config.replace_prob
            negative_prob = model_config.negative_prob

        train_df = df[df["user_id"].isin(train_users)]
        valid_df = df[df["user_id"].isin(valid_users)]
        test_df = df[df["user_id"].isin(test_users)]

        train_dataset = dataset(train_df, seq_len, num_skills, num_questions)
        valid_dataset = dataset(valid_df, seq_len, num_skills, num_questions)
        test_dataset = dataset(test_df, seq_len, num_skills, num_questions)

        print("train_ids", len(train_users))
        print("valid_ids", len(valid_users))
        print("test_ids", len(test_users))

        if "cl" in model_name:  # contrastive learning
            train_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        train_dataset,
                        seq_len,
                        mask_prob,
                        crop_prob,
                        permute_prob,
                        replace_prob,
                        negative_prob,
                        eval_mode=False,
                    ),
                    batch_size=batch_size,
                )
            )

            valid_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        valid_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )

            test_loader = accelerator.prepare(
                DataLoader(
                    SimCLRDatasetWrapper(
                        test_dataset, seq_len, 0, 0, 0, 0, 0, eval_mode=True
                    ),
                    batch_size=eval_batch_size,
                )
            )
        else:
            raise ValueError("This paper package supports only the CL4KT backbone")

        n_gpu = torch.cuda.device_count()
        if not distributed and n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate, weight_decay=model_config.l2)

        model, opt = accelerator.prepare(model, opt)

        test_auc, test_acc, test_rmse = model_train(
            fold,
            model,
            accelerator,
            opt,
            train_loader,
            valid_loader,
            test_loader,
            config,
            n_gpu,
        )

        test_aucs.append(test_auc)
        test_accs.append(test_acc)
        test_rmses.append(test_rmse)

    test_auc = np.mean(test_aucs)
    test_auc_std = np.std(test_aucs)
    test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    test_rmse = np.mean(test_rmses)
    test_rmse_std = np.std(test_rmses)

    now = (datetime.now() + timedelta(hours=9)).strftime("%Y%m%d-%H%M%S")  # KST time

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        log_dir_name = "official-split" if official_split else "5-fold-cv"
        log_out_path = os.path.join(os.path.join("logs", log_dir_name, "{}".format(data_name)))
        os.makedirs(log_out_path, exist_ok=True)
        with open(os.path.join(log_out_path, "{}-{}".format(model_name, now)), "w") as f:
            f.write("AUC\tACC\tRMSE\n")
            f.write("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))

        print("\nOfficial Split Result" if official_split else "\n5-fold CV Result")
        print("AUC\tACC\tRMSE")
        print("{:.5f}\t{:.5f}\t{:.5f}".format(test_auc, test_acc, test_rmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/paper.yaml",
        help="Base YAML configuration file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cl4kt",
        help="The name of the model to train. \
            This paper package supports cl4kt.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="algebra05",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--reg_cl",
        type=float,
        default=0.1,
        help="regularization parameter contrastive learning loss",
    )
    parser.add_argument("--mask_prob", type=float, default=0.2, help="mask probability")
    parser.add_argument("--crop_prob", type=float, default=0.3, help="crop probability")
    parser.add_argument(
        "--permute_prob", type=float, default=0.3, help="permute probability"
    )
    parser.add_argument(
        "--replace_prob", type=float, default=0.3, help="replace probability"
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=512, help="train batch size"
    )
    parser.add_argument("--l2", type=float, default=0.0, help="l2 regularization param")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU id or ids to make visible for training, e.g. 0 or 1",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Expect a torchrun launch and use Accelerator's DDP preparation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the YAML configuration.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="saved_model",
        help="Root directory used to save checkpoints for this run.",
    )
    parser.add_argument(
        "--log_path",
        default=None,
        help="Override train_config.log_path.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override train_config.num_epochs.",
    )
    parser.add_argument(
        "--use_joint_training_module",
        action="store_true",
        help="Enable cluster-friendly joint training loss for CL4KT.",
    )
    parser.add_argument(
        "--joint_training_losses",
        type=str,
        default="soft,kmeans,separation,temporal",
        help="Comma-separated joint losses: soft,kmeans,separation,temporal.",
    )
    args = parser.parse_args()

    base_cfg_file = PathManager.open(args.config, "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)
    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.checkpoint_dir = args.checkpoint_dir
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer
    if args.log_path is not None:
        cfg.train_config.log_path = args.log_path
    if args.num_epochs is not None:
        cfg.train_config.num_epochs = args.num_epochs

    if args.model_name == "cl4kt":
        cfg.cl4kt_config.reg_cl = args.reg_cl
        cfg.cl4kt_config.mask_prob = args.mask_prob
        cfg.cl4kt_config.crop_prob = args.crop_prob
        cfg.cl4kt_config.permute_prob = args.permute_prob
        cfg.cl4kt_config.replace_prob = args.replace_prob
        cfg.cl4kt_config.negative_prob = args.negative_prob
        cfg.cl4kt_config.dropout = args.dropout
        cfg.cl4kt_config.l2 = args.l2
        cfg.cl4kt_config.use_joint_training_module = args.use_joint_training_module
        cfg.cl4kt_config.joint_training_losses = args.joint_training_losses
        cfg.cl4kt_config.joint_cluster_k = 3
        cfg.cl4kt_config.joint_cluster_project_dim = cfg.cl4kt_config.hidden_size
        cfg.cl4kt_config.joint_cluster_loss_weight = 0.05
        cfg.cl4kt_config.joint_soft_loss_weight = 1.0
        cfg.cl4kt_config.joint_kmeans_loss_weight = 0.1
        cfg.cl4kt_config.joint_separation_loss_weight = 0.01
        cfg.cl4kt_config.joint_temporal_loss_weight = 0.01
        cfg.cl4kt_config.joint_student_t_alpha = 1.0
        cfg.cl4kt_config.joint_normalize = True
    else:
        raise ValueError("This paper package supports only --model_name cl4kt")

    cfg.freeze()

    print(cfg)
    main(cfg, args.gpu, args.ddp)
