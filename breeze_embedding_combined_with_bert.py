import gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

"""
目標：把 Breeze-7B 的字典與詞嵌入規則與 BERT 主要模型混合在一起，透過
"""

# Step1. 載入模型與斷詞器
breeze_tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")
breeze_model = AutoModel.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")
bert_model = AutoModel.from_pretrained("bert-base-chinese")

# Step2. 分別把 Breeze-7B 的詞嵌入、BERT 的 Encoder、Pooler 擷取出來
breeze_model_embedding = breeze_model.embed_tokens
bert_model_encoder = bert_model.encoder
bert_model_pooler = bert_model.pooler

# Note: 把不需要的模型刪除掉，以利節省空間
del breeze_model, bert_model; gc.collect()

# Step3. 補足 Breeze 至 BERT 之間的特徵維度差異
breeze_to_bert = nn.Linear(4096, 768)
bert_to_breeze = nn.Linear(768, 4096)

# Step4. 給定範例已知文字以及它的下一個文字
known_text = "敬請出席5月9日(四)下午2點2樓訓練教"
next_text = "室"

# Step5. 開始推論已知文字
# 最終結果的維度是 (1, 4096)
known_text_encoding = breeze_tokenizer(known_text, return_tensors = "pt")
breeze_model_embedding_outputs = breeze_model_embedding(known_text_encoding["input_ids"])
breeze_to_bert_outputs = breeze_to_bert(breeze_model_embedding_outputs)
bert_model_encoder_outputs = bert_model_encoder(breeze_to_bert_outputs)
bert_model_pooler_outputs = bert_model_pooler(bert_model_encoder_outputs[0])
outputs = bert_to_breeze(bert_model_pooler_outputs)

# 把由已知文字計算而得的特徵向量藉由 Breeze-7B 的詞嵌入模型轉換成詞嵌入向量
breeze_model_embedding_weights = breeze_model_embedding.weight
outputs_probability = torch.inner(outputs, breeze_model_embedding_weights)
outputs_token = breeze_tokenizer.decode(torch.argmax(outputs_probability[0])) # 預測時使用

# Q1. 在模型訓練時，要用 Embeddings 的差作為損失值，還是要用「文字分類」想法，把詞嵌入向量轉回成每個字被選的機率後再計算損失值？

# Step6. 把答案文字轉換成 One-Hot Encoding
next_text_id = breeze_tokenizer.encode("室")[1]
next_text_one_hot_encoding = [
    1 if i == next_text_id else 0 for i in range(breeze_model_embedding_weights.size()[0])
]
next_text_one_hot_encoding = torch.FloatTensor(next_text_one_hot_encoding)

# Step7. 計算損失函數
