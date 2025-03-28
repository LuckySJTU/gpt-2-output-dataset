import sys
sys.path.append('.')

from detector.dataset import Corpus, EncodedDataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import torch
import h5py
import os
from utils import load_config, seed_everything
from vq_models import get_model
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np

SEED = 1234
SAVE_DIR = "/home/yxwang/gpt-2-output-dataset/data/small/" 
ALL_DATASETS = [
    'webtext',
    'small-117M',  'small-117M-k40',  'small-117M-nucleus',
    'medium-345M', 'medium-345M-k40', 'medium-345M-nucleus',
    'large-762M',  'large-762M-k40',  'large-762M-nucleus',
    'xl-1542M',    'xl-1542M-k40',    'xl-1542M-nucleus'
]
mse_criterion = torch.nn.MSELoss()
cos_criterion = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def load_checkpoint(model, optimizer, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']

def main():
    seed_everything(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", default='example.yaml')
    parser.add_argument("--model_config", default='vectorquantize.yaml')
    parser.add_argument("--ckpt_dir", default='./exp')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--patience", type=int, default=0,
                        help='setting patience>0 will enable early stopping.')
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scheduler", action='store_true', 
                        help='ReduceLRonPlateau')
    args = parser.parse_args()
    data_dir = 'data'
    real_dataset = 'webtext'
    fake_dataset = 'small-117M'
    model_name = '/data1/public/hf/openai-community/gpt2'
    max_sequence_length = 128
    min_sequence_length = None
    epoch_size = None
    token_dropout = None
    seed = None
    device = 'cuda:0'
    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    real_train, real_valid, real_test = real_corpus.train, real_corpus.valid, real_corpus.test
    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)
    fake_train, fake_valid, fake_test = fake_corpus.train, fake_corpus.valid, fake_corpus.test
    Sampler = RandomSampler # just one gpu generating

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.max_len=config.n_ctx
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config,
            cache_dir=model_name,
        )
    
    vq_model = get_model(args.model_config)
    load_checkpoint(vq_model, None, os.path.join(args.ckpt_dir, 'best_checkpoint.pt'))

    train_dataset = EncodedDataset(real_train, fake_train, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=Sampler(train_dataset))
    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))
    test_dataset = EncodedDataset(real_test, fake_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=Sampler(test_dataset))

    model.to(device)
    model.eval()
    vq_model.to(device)
    vq_model.eval()
    votes=1
    loader = train_loader
    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading test data ... {v}')]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]
    
    classifications = []
    mse_scores = []
    cos_scores = []
    idx = 0

    with tqdm(records, desc='Test') as loop, torch.no_grad():
        for example in loop:
            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                output = model(input_ids=texts, attention_mask=masks, output_hidden_states=True)
                hidden_states = output[2][5].squeeze(dim=0)
                out, indices, cmt_loss = vq_model(hidden_states)
                classifications.append(labels.item())
                mse_scores.append(mse_criterion(out, hidden_states).item())
                cos_scores.append(cos_criterion(out, hidden_states).mean().item())
    df1 = pd.DataFrame({
        "Label": classifications,
        "Score": mse_scores
    })
    df2 = pd.DataFrame({
        "Label": classifications,
        "Score": cos_scores
    })
    # 小提琴图
    sns.violinplot(x="Label", y="Score", data=df1)
    plt.title("MSE Distribution by Label")
    plt.savefig('vq_detector/exp/0327_vq_webtext/mse_violinplot.png')
    
    plt.clf()
    sns.violinplot(x="Label", y="Score", data=df2)
    plt.title("CosSim Distribution by Label")
    plt.savefig("vq_detector/exp/0327_vq_webtext/cos_violinplot.png")

    # 输入数据
    # classifications = [0, 1, 1, 0, ...]  # 真实标签
    # scores = [0.1, 0.9, 0.8, 0.2, ...]   # 模型输出分数

    # ==== ROC + Youden's J ====
    fpr, tpr, roc_thresholds = roc_curve(classifications, mse_scores)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    best_roc_idx = np.argmax(youden_j)
    best_roc_threshold = roc_thresholds[best_roc_idx]
    print(f"[ROC] Best Threshold: {best_roc_threshold:.4f}")
    print(f"[ROC] TPR: {tpr[best_roc_idx]:.4f}, FPR: {fpr[best_roc_idx]:.4f}")
    print(f"[ROC] AUC: {roc_auc:.4f}")
    # 绘图保存 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.scatter(fpr[best_roc_idx], tpr[best_roc_idx], color='red', label=f'Best Threshold = {best_roc_threshold:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vq_detector/exp/0327_vq_webtext/mse_roc_curve.png')
    plt.close()

    fpr, tpr, roc_thresholds = roc_curve(classifications, cos_scores)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    best_roc_idx = np.argmax(youden_j)
    best_roc_threshold = roc_thresholds[best_roc_idx]
    print(f"[ROC] Best Threshold: {best_roc_threshold:.4f}")
    print(f"[ROC] TPR: {tpr[best_roc_idx]:.4f}, FPR: {fpr[best_roc_idx]:.4f}")
    print(f"[ROC] AUC: {roc_auc:.4f}")
    # 绘图保存 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    plt.scatter(fpr[best_roc_idx], tpr[best_roc_idx], color='red', label=f'Best Threshold = {best_roc_threshold:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vq_detector/exp/0327_vq_webtext/cos_roc_curve.png')
    plt.close()

    # ==== PR + F1 ====
    precision, recall, pr_thresholds = precision_recall_curve(classifications, mse_scores)
    pr_auc = average_precision_score(classifications, mse_scores)
    # 计算 F1 分数（注意阈值少一个）
    f1_scores = 2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
    best_pr_idx = np.argmax(f1_scores)
    best_pr_threshold = pr_thresholds[best_pr_idx]
    print(f"[PR] Best Threshold: {best_pr_threshold:.4f}")
    print(f"[PR] Precision: {precision[best_pr_idx+1]:.4f}, Recall: {recall[best_pr_idx+1]:.4f}, F1: {f1_scores[best_pr_idx]:.4f}")
    print(f"[PR] AUC (Average Precision): {pr_auc:.4f}")
    # 绘图保存 PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {pr_auc:.2f})')
    plt.scatter(recall[best_pr_idx+1], precision[best_pr_idx+1], color='red', label=f'Best Threshold = {best_pr_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vq_detector/exp/0327_vq_webtext/mse_pr_curve.png')
    plt.close()

    precision, recall, pr_thresholds = precision_recall_curve(classifications, cos_scores)
    pr_auc = average_precision_score(classifications, cos_scores)
    # 计算 F1 分数（注意阈值少一个）
    f1_scores = 2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
    best_pr_idx = np.argmax(f1_scores)
    best_pr_threshold = pr_thresholds[best_pr_idx]
    print(f"[PR] Best Threshold: {best_pr_threshold:.4f}")
    print(f"[PR] Precision: {precision[best_pr_idx+1]:.4f}, Recall: {recall[best_pr_idx+1]:.4f}, F1: {f1_scores[best_pr_idx]:.4f}")
    print(f"[PR] AUC (Average Precision): {pr_auc:.4f}")
    # 绘图保存 PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {pr_auc:.2f})')
    plt.scatter(recall[best_pr_idx+1], precision[best_pr_idx+1], color='red', label=f'Best Threshold = {best_pr_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('vq_detector/exp/0327_vq_webtext/cos_pr_curve.png')
    plt.close()

if __name__ == '__main__':
    main()