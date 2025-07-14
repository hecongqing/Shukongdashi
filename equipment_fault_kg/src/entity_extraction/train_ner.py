"""
故障NER模型训练脚本
"""

import argparse
import json
from pathlib import Path
from loguru import logger

from fault_ner_model import FaultNERTrainer


def main():
    parser = argparse.ArgumentParser(description="训练故障NER模型")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据路径")
    parser.add_argument("--val_data", type=str, help="验证数据路径，如果不提供则从训练集分割")
    parser.add_argument("--output_dir", type=str, required=True, help="模型输出目录")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="预训练模型名称")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "output_dir": args.output_dir
    }
    
    logger.info("开始训练故障NER模型")
    logger.info(f"配置: {json.dumps(config, ensure_ascii=False, indent=2)}")
    
    # 创建训练器
    trainer = FaultNERTrainer(config)
    
    # 训练
    trainer.train(args.train_data, args.val_data)
    
    logger.info("训练完成")


if __name__ == "__main__":
    main()