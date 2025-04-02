import sys
sys.path.append('.')

from detector.dataset import Corpus, EncodedDataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import torch
import h5py
import os

SAVE_DIR = "/home/yxwang/gpt-2-output-dataset/data/Meta-Llama-3-8B/" 
ALL_DATASETS = [
    'webtext',
    'small-117M',  'small-117M-k40',  'small-117M-nucleus',
    'medium-345M', 'medium-345M-k40', 'medium-345M-nucleus',
    'large-762M',  'large-762M-k40',  'large-762M-nucleus',
    'xl-1542M',    'xl-1542M-k40',    'xl-1542M-nucleus'
]

def main():
    data_dir = 'data'
    real_dataset = 'webtext'
    model_name = '/data1/public/hf/meta-llama/Meta-Llama-3-8B'
    max_sequence_length = 128
    min_sequence_length = None
    epoch_size = None
    token_dropout = None
    seed = None
    device = 'cuda:0'
    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    real_train, real_valid, real_test = real_corpus.train, real_corpus.valid, real_corpus.test
    Sampler = RandomSampler # just one gpu generating
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.max_len=config.max_position_embeddings
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=config,
            cache_dir=model_name,
        )
    train_dataset = EncodedDataset(real_train, [], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=Sampler(train_dataset))
    validation_dataset = EncodedDataset(real_valid, [], tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))
    test_dataset = EncodedDataset(real_test, [], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=Sampler(test_dataset))

    model.to(device)
    model.eval()
    votes=1
    records = [record for v in range(votes) for record in tqdm(test_loader, desc=f'Preloading test data ... {v}')]
    records = [[records[v * len(test_loader) + i] for v in range(votes)] for i in range(len(test_loader))]
    h5_file = h5py.File(os.path.join(SAVE_DIR, 'test.h5'), 'w')
    hidden_states_dataset = h5_file.create_dataset('hidden_states', shape=(sum(i[0][0].shape[1] for i in records), 4096), dtype='f4')
    idx = 0
    with tqdm(records, desc='Test') as loop, torch.no_grad():
        for example in loop:
            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                output = model(input_ids=texts, attention_mask=masks, output_hidden_states=True)
                hidden_states = output.hidden_states[16].squeeze(dim=0)
                hidden_states_dataset[idx:idx+hidden_states.shape[0]] = hidden_states.cpu().numpy()
                idx += hidden_states.shape[0]
    
    votes=1
    records = [record for v in range(votes) for record in tqdm(validation_loader, desc=f'Preloading valid data ... {v}')]
    records = [[records[v * len(validation_loader) + i] for v in range(votes)] for i in range(len(validation_loader))]
    h5_file = h5py.File(os.path.join(SAVE_DIR, 'valid.h5'), 'w')
    hidden_states_dataset = h5_file.create_dataset('hidden_states', shape=(sum(i[0][0].shape[1] for i in records), 4096), dtype='f4')
    idx = 0
    with tqdm(records, desc='Valid') as loop, torch.no_grad():
        for example in loop:
            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                output = model(input_ids=texts, attention_mask=masks, output_hidden_states=True)
                hidden_states = output.hidden_states[16].squeeze(dim=0)
                hidden_states_dataset[idx:idx+hidden_states.shape[0]] = hidden_states.cpu().numpy()
                idx += hidden_states.shape[0]

    votes=1
    records = [record for v in range(votes) for record in tqdm(train_loader, desc=f'Preloading train data ... {v}')]
    records = [[records[v * len(train_loader) + i] for v in range(votes)] for i in range(len(train_loader))]
    h5_file = h5py.File(os.path.join(SAVE_DIR, 'train.h5'), 'w')
    hidden_states_dataset = h5_file.create_dataset('hidden_states', shape=(sum(i[0][0].shape[1] for i in records), 4096), dtype='f4')
    idx = 0
    with tqdm(records, desc='Train') as loop, torch.no_grad():
        for example in loop:
            for texts, masks, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                output = model(input_ids=texts, attention_mask=masks, output_hidden_states=True)
                hidden_states = output.hidden_states[16].squeeze(dim=0)
                hidden_states_dataset[idx:idx+hidden_states.shape[0]] = hidden_states.cpu().numpy()
                idx += hidden_states.shape[0]


if __name__ == '__main__':
    main()