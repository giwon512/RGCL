# from tqdm import tqdm

# for i in tqdm(range(100)):
#     # Some long-running task
#     pass

from transformers import BertTokenizer
from transformers import BertModel

# 토크나이저 및 텍스트 토큰화 설정
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "BERT 모델의 tokenizer 예시입니다."
tokens = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

print(tokens)  # Tokens 형태 확인

# BERT 모델 로드 및 출력 hidden_states 설정
bert = BertModel.from_pretrained('bert-base-uncased')
bert.config.output_hidden_states = True

# 입력 텐서를 모델에 전달하고 출력 받기
output = bert(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])

# 출력 형태 확인
print(output.last_hidden_state)
