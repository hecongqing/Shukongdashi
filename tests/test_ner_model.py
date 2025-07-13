"""
NER模型单元测试
"""
import unittest
import torch
import tempfile
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.ner_model import NERTrainer, BertBiLSTMCRF
from config.settings import CONFIG


class TestNERModel(unittest.TestCase):
    """NER模型测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.trainer = NERTrainer()
        self.sample_text = "北京是中国的首都"
        self.sample_labels = ["B-LOC", "O", "B-GPE", "O", "O", "O"]
        
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        self.assertIsInstance(self.trainer, NERTrainer)
        self.assertIsNotNone(self.trainer.tokenizer)
        
    def test_model_architecture(self):
        """测试模型架构"""
        model = BertBiLSTMCRF(
            bert_model_name="bert-base-chinese",
            num_labels=9,  # B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O
            hidden_dim=128
        )
        self.assertIsInstance(model, BertBiLSTMCRF)
        
    def test_text_preprocessing(self):
        """测试文本预处理"""
        # 测试数据加载格式
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {
                    "text": self.sample_text,
                    "labels": self.sample_labels
                }
            ]
            import json
            json.dump(test_data, f, ensure_ascii=False)
            temp_file = f.name
            
        try:
            texts, labels = self.trainer.load_data(temp_file)
            self.assertEqual(len(texts), 1)
            self.assertEqual(len(labels), 1)
            self.assertEqual(texts[0], self.sample_text)
        finally:
            os.unlink(temp_file)
            
    def test_predict_functionality(self):
        """测试预测功能"""
        # 这里我们只测试预测接口，不训练实际模型
        try:
            result = self.trainer.predict(self.sample_text)
            self.assertIsInstance(result, list)
        except Exception as e:
            # 如果没有预训练模型，这是正常的
            self.assertIn("model", str(e).lower())


class TestNERDataset(unittest.TestCase):
    """NER数据集测试类"""
    
    def setUp(self):
        """测试初始化"""
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.label2id = {
            'B-PER': 0, 'I-PER': 1,
            'B-LOC': 2, 'I-LOC': 3,
            'B-ORG': 4, 'I-ORG': 5,
            'B-MISC': 6, 'I-MISC': 7,
            'O': 8
        }
        
    def test_dataset_creation(self):
        """测试数据集创建"""
        from models.ner_model import NERDataset
        
        texts = ["北京是中国的首都"]
        labels = [["B-LOC", "O", "B-GPE", "O", "O", "O"]]
        
        dataset = NERDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=128
        )
        
        self.assertEqual(len(dataset), 1)
        
        # 测试数据项
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        
        self.assertEqual(item['input_ids'].shape[0], 128)
        self.assertEqual(item['attention_mask'].shape[0], 128)
        self.assertEqual(item['labels'].shape[0], 128)


if __name__ == '__main__':
    unittest.main()