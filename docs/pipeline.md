# Pipeline

## 개요

현재 프로젝트는 한 개의 중심 node에 대한 다음 시점 수요를 예측하는 patch 기반 DMVST 모델입니다. 구조는 크게 세 부분으로 나뉩니다.

1. Dataset / split
   raw grid와 기상 데이터를 sample 단위 `(patch, 중심 node)`로 변환합니다.
2. Forecasting model
   neural branch와 retrieval branch를 만든 뒤 gate로 합칩니다.
3. Evaluation
   MAE, MAPE, RMSE와 top-k node metric을 계산하고 시각화를 저장합니다.

이 문서는 코드 기준으로 모듈별 역할, 주요 변수, shape, 디버깅 포인트를 설명합니다.

## 전체 흐름

엔트리포인트는 [main.py](/home/jinsu/PycharmProjects/DMVST/main.py)입니다.

실행 순서는 아래와 같습니다.

1. `DMVSTDataset(**config.dataset)` 생성
2. `dataset_size = len(dataset)` 계산
3. `train_end`, `valid_end`를 `num_nodes` 배수로 맞춰 train/valid/test 경계 정렬
4. `warmup_steps = model.IRModule.k * dataset.num_nodes`
5. `dataset.get_train_graph_path(train_end)`로 train prefix 전용 LINE graph csv 준비
6. `IRModule(dataset, device, k)` 생성
7. `DMVST(..., line_graph_path=..., ir_module=...)` 생성
8. `train_dataset = Subset(dataset, range(warmup_steps, train_end))`
9. `valid_dataset = Subset(dataset, range(train_end, valid_end))`
10. Hugging Face `Trainer`로 학습 및 validation metric 계산
11. `test_dataset = Subset(dataset, range(valid_end, dataset_size))`
12. `test_loop(...)`로 최종 test 평가 및 결과 저장

중요한 점은 retrieval과 LINE이 모두 train 정보만 보도록 제약된다는 것입니다.

- LINE graph는 `train_end`까지의 raw demand로만 생성됩니다.
- IR retrieval은 full dataset을 DB로 들고 있지만, 실제 검색은 항상 `sample_idx`보다 이전 시점 prefix만 허용합니다.

## Dataset

코드는 [dmvst_dataset.py](/home/jinsu/PycharmProjects/DMVST/dataset_struct/dmvst_dataset.py)에 있습니다.

### 주요 멤버 변수

`self.raw_demands`
: 원본 수요 tensor, shape `(T, X, Y)`

`self.demands`
: patch 추출을 위해 spatial zero padding이 적용된 tensor, shape `(T, X + pad*2, Y + pad*2)`

`self.T, self.X, self.Y`
: 시간 길이와 grid 크기

`self.num_nodes`
: `X * Y`

`self.patch_size`
: 한 sample이 보는 spatial patch 한 변 길이

`self.time_step`
: 한 sample이 보는 temporal 길이

`self.temporal_features`
: 기상 feature tensor, shape `(T, F)`

### sample 생성 규칙

`__getitem__(idx)`는 `idx`를 time-major order로 해석합니다.

```text
t_idx = idx // num_nodes
xy_idx = idx % num_nodes
x_idx = xy_idx // Y
y_idx = xy_idx % Y
```

즉 sample 순서는 다음과 같습니다.

- 같은 시간 `t_idx`에서 모든 node를 순회
- 그 다음 시간으로 이동

이 순서 때문에 `sample_idx // num_nodes`는 현재 sample의 "실제 시점 index"와 동일하게 쓸 수 있습니다. retrieval에서 causal prefix를 제한할 때 이 값을 그대로 사용합니다.

### 반환되는 sample

`DMVSTDataset.__getitem__`은 아래 dict를 반환합니다.

`demands`
: `(time_step, patch_size, patch_size)`  
`self.demands[t_idx:t_idx + time_step, x_idx:x_idx + patch_size, y_idx:y_idx + patch_size]`

`labels`
: scalar  
`self.raw_demands[t_idx + time_step, x_idx, y_idx]`

`temporal_features`
: `(time_step, num_temporal_features)`  
현재 입력 구간의 기상 feature

`node_id`
: scalar  
`x_idx * Y + y_idx`

`sample_idx`
: scalar  
absolute dataset index. retrieval causal mask의 기준

### collate_fn 출력

`collate_fn`은 batch를 아래 shape로 묶습니다.

`demands`
: `(B, time_step, patch_size, patch_size)`

`labels`
: `(B,)`

`temporal_features`
: `(B, time_step, F)`

`node_ids`
: `(B,)`

`sample_idx`
: `(B,)`

### get_train_graph_path

`get_train_graph_path(num_samples)`는 LINE graph를 train prefix 기준으로 생성하거나 기존 파일을 재사용합니다.

중요 변수:

`train_label_steps`
: train sample 수를 시간 step 수로 바꾼 값

`raw_steps`
: 실제 graph 생성에 사용할 raw demand 길이  
`train_label_steps + time_step`

출력 파일:

```text
data/processed/dmvst_graph_edges_{grid_size}_train_{raw_steps}.csv
```

디버깅 포인트:

- train split이 바뀌면 graph 파일명도 바뀌어야 합니다.
- train-only graph가 아니라 full graph가 로드되고 있으면 `line_graph_path` 연결을 확인하면 됩니다.

## IRModule

코드는 [DMVSTModel.py](/home/jinsu/PycharmProjects/DMVST/models/DMVSTModel.py)에 있습니다.

### 역할

`IRModule`은 현재 batch의 patch query를 받아, 같은 중심 위치의 과거 patch window만 cosine similarity로 검색하고 retrieval prediction scalar를 반환합니다.

참고로 이 구현은 `CustomDMVST`의 full-region retrieval을 현재 프로젝트에 맞춰 patch-aware 방식으로 바꾼 버전입니다.

### 초기화 시 만들어지는 DB

`IRModule.__init__(dataset, device, k, storage_dtype=torch.float16)`

주요 변수:

`self.num_nodes`
: 전체 중심 위치 개수

`self.num_time_samples`
: `T - time_step`

`flat_dim`
: `time_step * patch_size * patch_size`

`self.db_keys`
: shape `(num_nodes, num_time_samples, flat_dim)`  
각 중심 위치별 과거 patch window DB. 메모리 절감을 위해 기본 `float16`

`self.db_norms`
: shape `(num_nodes, num_time_samples)`  
cosine normalization용 norm

`self.db_values`
: shape `(num_nodes, num_time_samples)`  
각 중심 위치의 다음 시점 label scalar

DB 구축 방식:

1. `node_id`마다 중심 위치를 하나 고릅니다.
2. padded demand tensor에서 해당 위치의 patch 시계열 `(T, patch, patch)`를 자릅니다.
3. time 축 `unfold(0, time_step, 1)`로 sliding window를 만듭니다.
4. 각 window를 flatten해서 `db_keys[node_id]`에 저장합니다.
5. 같은 위치의 실제 label은 `db_values[node_id]`에 저장합니다.

중요한 구현 포인트:

- `patch_windows = patch_series.unfold(0, time_step, 1)[:self.num_time_samples]`
- 여기서 `[:self.num_time_samples]`를 빼면 query 수보다 window가 1개 많아져 shape mismatch가 납니다.

### forward 입력과 출력

입력:

`query_demands`
: `(B, time_step, patch_size, patch_size)`

`node_ids`
: `(B,)`

`sample_idx`
: `(B,)`

출력:

`aggregated`
: `(B,)` retrieval prediction

`retrieved_indices`
: `(B, k)` prefix 내부에서 선택된 index, 없는 자리는 `-1`

### 검색 로직

`forward` 내부 흐름은 아래와 같습니다.

1. `query_demands.flatten(start_dim=1)`  
   query를 `(B, flat_dim)`로 변환
2. `time_indices = sample_idx // num_nodes`  
   각 sample의 시점 index 계산
3. batch 안에서 `node_id`별로 group 생성
4. 같은 `node_id` group에 대해서만 같은 DB slice 사용
5. candidate prefix를 `0..time_idx-1`로 제한
6. cosine similarity 계산
7. top-k 선택
8. softmax weight로 label 가중합

핵심 제약:

- 검색 대상은 같은 `node_id`만 허용
- 미래 시점은 절대 검색 불가

디버깅 포인트:

- `sample_idx`가 batch에 없으면 causal retrieval이 깨집니다.
- `node_ids`가 잘못되면 엉뚱한 위치 patch를 검색합니다.
- 첫 시점 근처에서는 candidate가 없어서 retrieval output이 0이 될 수 있습니다.
- `warmup_steps = k * num_nodes`를 쓰는 이유는 node별 최소 `k`개 시점 prefix를 보장하기 위해서입니다.

## LINE graph와 embedding

`line(...)` 함수는 graph csv를 읽어 `cogdl`의 LINE 임베딩을 생성합니다.

입력 파일:

- 기본값은 `data/processed/dmvst_graph_edges_{grid_size}.csv`
- 현재 main에서는 `line_graph_path = dataset.get_train_graph_path(train_end)`를 넘기므로 train prefix 전용 graph를 사용합니다.

주요 변수:

`edges`
: csv에서 읽은 `(u, v, w)`

`node_to_id`
: graph node id를 연속 index로 매핑

`edge_index`
: shape `(2, E)`

`edge_weight`
: shape `(E,)`

출력:

`line_embeddings`
: `(num_nodes, Line.dimension)`

디버깅 포인트:

- train-only graph가 맞는지는 log의 `Loading LINE embeddings from graph CSV:` 경로를 보면 확인할 수 있습니다.
- `num_nodes`가 dataset node 수와 맞지 않으면 `node_ids` indexing에서 바로 오류가 납니다.

## LocalCNN

역할:

- 각 시간 step의 patch를 CNN으로 임베딩

입력:

`x`
: `(B, time_step, patch_size, patch_size)`

처리:

1. `x.view(B * time_step, 1, width, height)`
2. Conv2d + ReLU stack
3. Flatten
4. Linear projection
5. `(B, time_step, demand_embedding_dim)`로 복원

주요 변수:

`channels`
: conv block을 거치며 증가

`embedding_layer`
: spatial conv 출력을 최종 demand embedding dimension으로 압축

디버깅 포인트:

- `Local_cnn.neighborhood_size`는 반드시 dataset `patch_size`와 같아야 합니다.
- 둘이 다르면 `embedding_layer` 입력 차원 mismatch가 발생합니다.

## DMVST model

`DMVST`는 neural branch와 retrieval branch를 fuse하는 최종 모델입니다.

### 생성자에서 만들어지는 모듈

`temporal_layer`
: `(num_temporal_features -> temporal_embedding_dim)` linear

`local_cnn`
: patch encoder

`line_embeddings`
: graph 기반 node embedding table

`context_embedding_layer`
: `Line.dimension -> context_embedding_dim`

`lstm`
: 입력은 `demand_embedding_dim + temporal_embedding_dim`

`final_fc`
: neural branch scalar prediction head

`lambda_layer`
: retrieval가 켜진 경우에만 생성되는 fusion gate head

### predict 입력

`demands`
: `(B, time_step, patch_size, patch_size)`

`temporal_features`
: `(B, time_step, F)`

`node_ids`
: `(B,)`

`sample_idx`
: `(B,)`

### predict 내부 흐름

1. `demands_features = local_cnn(demands)`  
   shape `(B, time_step, demand_embedding_dim)`
2. `temporal_emb = temporal_layer(temporal_features)`  
   shape `(B, time_step, temporal_embedding_dim)`
3. `context_emb = context_embedding_layer(line_embeddings[node_ids])`  
   shape `(B, context_embedding_dim)`
4. `lstm_input = cat([demands_features, temporal_emb], dim=-1)`
5. `lstm_out[:, -1, :]`로 마지막 시간 hidden 추출
6. `final_features = cat([last_hidden, context_emb], dim=-1)`  
   shape `(B, fusion_feature_dim)`
7. `output = final_fc(final_features)`  
   neural branch prediction
8. IRModule이 있으면 `ir_out = ir_module(demands, node_ids, sample_idx)`
9. `lambda_weight = sigmoid(lambda_layer(cat([final_features, ir_out], dim=-1)))`
10. `lambda * output + (1 - lambda) * ir_out`

### forward

`forward(...)`는 Hugging Face `Trainer` 호환 wrapper입니다.

반환 dict:

`predictions`
: `(B,)`

`loss`
: labels가 있을 때만 추가

디버깅 포인트:

- IRModule이 켜져 있는데 `sample_idx`를 넘기지 않으면 `ValueError`가 납니다.
- `line_embeddings[node_ids]`에서 오류가 나면 dataset `num_nodes`와 graph node 수를 먼저 확인하면 됩니다.
- retrieval가 너무 강하게 작동하면 `lambda_weight`를 로그로 뽑아 neural branch 대비 gate 비중을 보면 됩니다.

## Loss

코드는 [dmvst_loss.py](/home/jinsu/PycharmProjects/DMVST/loss_fn/dmvst_loss.py)에 있습니다.

손실식:

```text
diff = y_true - y_pred
term1 = diff^2
term2 = (diff / (y_true + eps))^2
loss = term1 + gamma * term2
```

의미:

- `term1`: 절대 오차 기반 제곱 손실
- `term2`: 상대 오차 기반 제곱 손실
- `gamma`: 상대 오차 항 가중치
- `eps`: 분모 안정화 상수

디버깅 포인트:

- 작은 label에서 loss가 과도하게 커지면 `eps`와 `gamma`를 먼저 확인하면 됩니다.
- `reduction='mean'`이 기본이므로 batch 크기에 따라 loss scale이 크게 흔들리지는 않습니다.

## Split과 학습 루프

`main.py` 기준 주요 변수:

`dataset_size`
: 전체 sample 수

`train_end`
: train 종료 index. `num_nodes` 배수로 내림해서 시간 경계 정렬

`valid_end`
: validation 종료 index. `num_nodes` 배수로 내림해서 시간 경계 정렬

`ir_config`
: `config.model.IRModule`

`warmup_steps`
: `ir_config.k * dataset.num_nodes`

`train_indices`
: `[warmup_steps, train_end)`

`valid_indices`
: `[train_end, valid_end)`

`test_indices`
: `[valid_end, dataset_size)`

디버깅 포인트:

- `train_end <= warmup_steps`이면 train sample이 사라지므로 바로 `ValueError`를 냅니다.
- `valid_end <= train_end`이면 valid sample이 사라집니다.
- `valid_end >= dataset_size`이면 test sample이 사라집니다.
- `split.train_ratio`가 작거나 `IRModule.k`가 너무 크면 첫 조건에 걸립니다.
- `remove_unused_columns=false`가 꺼지면 `sample_idx`가 Trainer에서 drop될 수 있으므로 유지해야 합니다.

## Evaluation

코드는 [test.py](/home/jinsu/PycharmProjects/DMVST/runners/test.py)에 있습니다.

Hugging Face `Trainer`의 `compute_metrics`는 validation split에서 계산됩니다. 아래 `test_loop`는 학습 종료 후 별도 test split에만 적용됩니다.

### test_loop

입력 batch에서 다음을 꺼냅니다.

- `demands`
- `temporal_features`
- `node_ids`
- `sample_idx`
- `labels`

호출:

```python
outputs = model.predict(demands, temporal_features, node_ids, sample_idx)
outputs = torch.clamp(outputs, min=0.0)
```

계산 지표:

`MAE`
: `L1Loss`

`MAPE`
: `mean(abs(label - pred) / (label + 1)) * 100`

`RMSE`
: `sqrt(MSE)`

`Top{k}_MAE`, `Top{k}_MAPE`
: test 구간에서 총수요가 큰 node만 따로 계산한 metric

### topk_node_loss

동작:

1. `pred`, `label`을 `(-1, total_nodes)`로 reshape
2. node별 label 합 계산
3. 상위 `k` node index 선택
4. 해당 node만 잘라 metric 계산

주의:

- 이 함수는 evaluation용 `test.k`를 사용합니다.
- retrieval top-k인 `model.IRModule.k`와는 다른 값입니다.

### 시각화

`visualize_predictions`는 다음 파일을 저장합니다.

- `demand_error_analysis.png`
- `predictions_max_demand_node.png`
- `predictions_min_demand_node.png`
- `predictions_mid_demand_node.png`

디버깅 포인트:

- reshape 전제가 깨지면 `test_dataset`이 전체 eval 구간의 연속 node-major block인지 확인해야 합니다.
- 현재 split은 `num_nodes` 배수로 정렬하므로 eval 구간 reshape가 가능합니다.

## 설정 파일 설명

기준 파일은 [config.yaml](/home/jinsu/PycharmProjects/DMVST/configs/config.yaml)입니다.

### dataset

`time_step`
: 입력 시계열 길이

`patch_size`
: spatial patch 크기

`grid_size`
: 불러올 `grid(size).npy`의 size

### criterion

`gamma`
: relative loss weight

`eps`
: relative loss denominator stabilizer

### model

`Local_cnn`
: patch encoder 설정

`LSTM`
: temporal model 설정

`Line`
: graph embedding 설정

`IRModule.k`
: retrieval top-k. warmup 길이 계산에도 사용

`demand_embedding_dim`
: CNN output embedding dim

`temporal_embedding_dim`
: weather embedding dim

`context_embedding_dim`
: LINE embedding projected dim

`num_temporal_features`
: 기상 feature 수

### test

`k`
: top-k evaluation metric용 node 수

## 디버깅 체크리스트

1. shape mismatch가 나면 먼저 `demands`, `temporal_features`, `node_ids`, `sample_idx` shape를 확인합니다.
2. retrieval 결과가 전부 0이면 `sample_idx // num_nodes`가 너무 작거나 `warmup_steps`가 충분하지 않은지 봅니다.
3. LINE 관련 index error가 나면 train graph csv와 dataset node 수가 같은지 확인합니다.
4. eval reshape 오류가 나면 `train_end`가 `num_nodes` 배수로 맞춰졌는지 확인합니다.
5. loss가 폭증하면 `criterion.gamma`, `criterion.eps`, label scale을 같이 봅니다.
6. 예측이 모두 비슷하면 `lambda_weight`가 retrieval 쪽으로 과도하게 치우쳤는지 확인합니다.
