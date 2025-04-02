import os
import sys
import argparse
from transformers import AutoConfig, Trainer, TrainingArguments, pipeline, AutoTokenizer, TrainerCallback
from vq_gpt2_vqvae_all_layers import VQVAEGPT2Config, VQVAEGPT2, load_dataset, collect_fn
from utils import load_config
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 1. 加载数据集
def load_data(prefix, tokenizer):
    logging.info(f'Dataset from {prefix}')
    dataset = load_dataset(prefix, tokenizer)
    return dataset

# 2. 配置GPT-2模型和Tokenizer
def load_model(model_name="gpt2", model_config_path='conf/model_config.yaml'):
    gpt2_config = AutoConfig.from_pretrained(model_name)
    vq_config = load_config(model_config_path)
    hlmgpt2_config = VQVAEGPT2Config(gpt2_config, model_name, vq_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VQVAEGPT2(hlmgpt2_config)
    tokenizer.max_len = gpt2_config.n_ctx
    return model, tokenizer

# 3. 数据预处理：tokenize数据集
def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_tensors='pt', padding=True, truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets

# 4. 配置训练参数
class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_metrics = {}
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        self._memory_metrics["rec_loss"] = outputs['rec_loss'].item()
        self._memory_metrics["cmt_loss"] = outputs['cmt_loss'].item()
        return (loss, outputs) if return_outputs else loss

class MergeCustomMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(self, 'trainer') and hasattr(self.trainer, '_memory_metrics'):
            logs["rec_loss"] = round(self.trainer._memory_metrics['rec_loss'], 4)
            logs["cmt_loss"] = round(self.trainer._memory_metrics['cmt_loss'], 4)
    def set_trainer(self, trainer):
        # Trainer调用此方法传递自身引用
        # 妙啊
        self.trainer = trainer

def configure_training(model, train_config, train_dataset, val_dataset):
    training_args = TrainingArguments(**train_config)
    custom_callback = MergeCustomMetricsCallback()
    trainer = MyTrainer(
        model=model,                        # 要训练的模型
        args=training_args,                 # 训练参数
        train_dataset=train_dataset,   # 训练数据集
        eval_dataset=val_dataset,       # 验证数据集
        data_collator=collect_fn,
        callbacks=[custom_callback],
    )
    custom_callback.set_trainer(trainer)
    return trainer

# 5. 训练模型
def train_model(trainer):
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)
    trainer.evaluate()

# 6. 保存模型
# def save_model(model):
#     model.save_pretrained("./gpt2_finetuned")

# 7. 生成文本
def generate_text(tokenizer):
    generator = pipeline("text-generation", model="./gpt2_finetuned", tokenizer=tokenizer)
    generated_text = generator("This is a test", max_length=50)
    print(generated_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vq_dir", default=None, help='Path to your vq model and config folder.')
    parser.add_argument("--data_config", default=None, help='Path to your vq data config file. If specified, will cover data_config in `vq_dir`, else will be `vq_dir/data_config.yaml`. At least one of --vq_dir or --data_config is required.')
    parser.add_argument("--model_config", default=None, help='Path to your vq model config file. If specified, will cover model_config in `vq_dir`, else will be `vq_dir/model_config.yaml`. At least one of --vq_dir or --data_config is required.')
    parser.add_argument("--vq_model", default=None, help='Path to your vq model checkpoint. If specified, will cover best_checkpoint in `vq_dir`, else will be `vq_dir/best_checkpoint.pt`. At least one of --vq_dir or --vq_model is required.')
    parser.add_argument("--model_name_or_path", default="/data1/public/hf/openai-community/gpt2", help='Path to gpt2 model')
    parser.add_argument("--train_config", default=None, help="Path to your train config file. If not specified, will be `vq_dir/train_config.yaml`. At least one of --vq_dir or --train_config is required.")
    parser.add_argument("--output_dir", default=None, help='Path to save model and logs. If not specified, will be `vq_dir_hlm`.')
    parser.add_argument("--ckpt_dir", default=None, help='Path to load model for test. If not specified, will use `--output_dir`')
    parser.add_argument("--test", action='store_true', help='Whether to run in test mode.')
    args = parser.parse_args()

    # 0. 处理参数
    if args.vq_dir is None:
        assert args.data_config is not None and args.model_config is not None and args.vq_model is not None and args.output_dir is not None and args.train_config is not None, 'If you dont use --vq_dir, you must specify --data_config, --model_config, --vq_model, --output_dir and --train_config'
    if args.data_config is None:
        args.data_config = os.path.join(args.vq_dir, 'data_config.yaml')
    if args.model_config is None:
        args.model_config = os.path.join(args.vq_dir, 'model_config.yaml')
    if args.train_config is None:
        args.train_config = os.path.join(args.vq_dir, 'train_config.yaml')
    if args.vq_model is None:
        args.vq_model = os.path.join(args.vq_dir, 'best_checkpoint.pt')
    if args.output_dir is None:
        args.output_dir = args.vq_dir
    assert os.path.exists(args.data_config), 'Please check your data_config file path'
    assert os.path.exists(args.model_config), 'Please check your model_config file path'
    # this parameter is not used in this script
    # assert os.path.exists(args.vq_model), 'Please check your vq_model file path'
    assert os.path.exists(args.train_config), 'Please check your train_config file path'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f'Creating output directory {args.output_dir}')
    else:
        logging.warning(f"Output directory {args.output_dir} already exists. May overwrite the existing files.")

    # 2. 加载GPT-2模型和Tokenizer
    logging.info('Loading gpt2 model...')
    model_name = args.model_name_or_path
    model, tokenizer = load_model(model_name, model_config_path=args.model_config)

    # 1. 加载数据
    logging.info('Loading data...')
    data_config = load_config(config_path=args.data_config)
    data_path = data_config['h5_file_path']
    dataset = load_data(data_path, tokenizer)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    # 3. 数据预处理
    logging.info('Tokenizing data')
    # 其实没有什么tokenize的步骤，只不过gpt将它写出来了而已
    # tokenized_datasets = tokenize_data(dataset, tokenizer)
    
    # 4. 配置训练参数
    logging.info('Configuring training...')
    train_config = load_config(config_path=args.train_config)
    if train_config.get('output_dir') is None:
        train_config['output_dir'] = args.output_dir
    if train_config.get('logging_dir') is None:
        train_config['logging_dir'] = train_config['output_dir']
    trainer = configure_training(model, train_config, train_dataset, val_dataset)
    
    if not args.test:
        # 5. 训练模型
        logging.info('Training model...')
        train_model(trainer)
    
        # 6. 保存模型
        logging.info('Saving model...')
        model.save_pretrained(args.output_dir)
    
    # 7. 测试模型
    logging.info('Testing model...')
    model = VQVAEGPT2.from_pretrained(args.ckpt_dir if args.ckpt_dir is not None else args.output_dir)
    trainer = configure_training(model, train_config, train_dataset, val_dataset)
    eval_metric = trainer.evaluate(test_dataset, metric_key_prefix='test')
    logging.info(f'Evaluation metric: {eval_metric}')

    # # 8. 其他测试标准，--test only
    # DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # # if args.test:
    # if True:
    #     logging.info("Validation for other metrics...")
    #     model.to(DEVICE)
    #     model.eval()
    #     with torch.no_grad():
    #         correct = [0]*NUM_QUANTIZER
    #         total = [0]*NUM_QUANTIZER
    #         # losses = []
    #         for batch in tqdm(test_dataset):
    #             # labels: 1*1024*num_quantizer
    #             # output: num_quantizer*(1*1024*codebooksize), list
    #             input_ids = torch.tensor(batch['input_ids']).to(DEVICE)
    #             input_ids = input_ids.unsqueeze(0)
    #             labels = torch.tensor(batch['label']).to(DEVICE)
    #             labels = labels.unsqueeze(0)
    #             output = model(input_ids=input_ids, labels=labels)
    #             # loss = output.loss
    #             # losses.append(loss.item())
    #             output = output.logits
    #             for i in range(len(output)):
    #                 # 计算每个码本的预测准确率
    #                 prediction = torch.argmax(output[i], dim=-1).squeeze()
    #                 target = labels[0,:,i]
    #                 correct[i] += torch.sum(prediction == target).item()
    #                 total[i] += target.shape[0]
    #     logging.info(f'Codebook prediction accuracy: {sum(correct) / sum(total):.4f}')
    #     logging.info(f'Pred acc on each codebook: {[correct[i]/total[i] for i in range(len(correct))]}')
    #     logging.info(f"Number of quantizer: {NUM_QUANTIZER}")
    #     # logging.info(f"Test Loss: {sum(losses) / len(losses)}")

if __name__ == "__main__":
    main()
