# DMVST

현재 브랜치는 `DMVST + LINE + patch-aware IR retrieval` 구조로 동작합니다. 기본 neural forecasting branch 위에, 같은 중심 위치의 과거 patch window만 검색하는 causal retrieval branch를 추가한 형태입니다.

상세 파이프라인과 디버깅용 변수 설명은 [docs/pipeline.md](/home/jinsu/PycharmProjects/DMVST/docs/pipeline.md)에 정리했습니다.

## 모델 개요

모델은 두 경로를 합칩니다.

1. Neural branch
   `LocalCNN`으로 `(time_step, patch, patch)` 수요 patch를 임베딩하고, 기상 feature와 concat한 뒤 LSTM으로 다음 시점 중심 수요를 예측합니다.
2. Retrieval branch
   현재 sample의 patch window를 query로 사용하고, 같은 `node_id`의 과거 patch window만 cosine similarity로 검색해 retrieval prediction을 만듭니다.

최종 출력은 gate 기반 fusion입니다.

- neural output: `final_fc(final_features)`
- retrieval output: `ir_out`
- final output: `lambda * neural + (1 - lambda) * ir_out`

## 핵심 특징

- 입력 sample은 전체 grid가 아니라 중심 node 기준 local patch 하나입니다.
- `IRModule`은 모든 지역을 검색하지 않고, 현재 sample과 같은 중심 위치의 과거 patch만 검색합니다.
- retrieval은 causal합니다. 현재 sample보다 이전 시점만 candidate로 사용합니다.
- `LINE` 그래프는 전체 시계열이 아니라 train split까지만 사용해 생성합니다.
- `test.k`는 evaluation용 top-k metric이고, `model.IRModule.k`는 retrieval top-k 및 warmup 길이 기준입니다. 둘은 다른 의미입니다.

## 데이터와 split

`DMVSTDataset`은 `grid(size).npy`와 `meteorological_data.csv`를 읽어 sample을 만듭니다.

각 sample은 다음으로 구성됩니다.

- `demands`: `(time_step, patch_size, patch_size)`
- `labels`: 다음 시점 중심 cell 수요 scalar
- `temporal_features`: `(time_step, num_temporal_features)`
- `node_id`: 중심 cell의 flatten index
- `sample_idx`: dataset 전체에서의 absolute sample index

학습 split은 `main.py`에서 다음 순서로 나뉩니다.

- `train_end = floor(split.train_ratio * len(dataset) / num_nodes) * num_nodes`
- `valid_end = floor((split.train_ratio + split.valid_ratio) * len(dataset) / num_nodes) * num_nodes`
- `warmup_steps = model.IRModule.k * num_nodes`
- retrieval-only prefix: `[0, warmup_steps)`
- train: `[warmup_steps, train_end)`
- valid: `[train_end, valid_end)`
- test: `[valid_end, len(dataset))`

앞의 `warmup_steps` 구간은 학습에는 사용하지 않고 retrieval candidate prefix 확보용으로만 남깁니다.
`Trainer.compute_metrics`는 valid split에서 계산되고, `test_loop`는 마지막 test split에만 적용됩니다.

## 실행 방법

기본 실행:

```bash
python main.py --config-name config
```

7000 grid 설정:

```bash
python main.py --config-name config7000
```

## 자주 바꾸는 인자

`dataset.time_step`
: 입력 시계열 길이

`dataset.patch_size`
: 각 sample이 보는 local patch 크기

`model.IRModule.k`
: retrieval top-k 크기. 동시에 retrieval-only warmup 길이 계산에 사용됩니다.

`test.k`
: 평가 시 상위 수요 node metric 계산용 top-k

`split.train_ratio`
: 전체 sample 중 train 종료 위치 비율

`split.valid_ratio`
: validation 구간 비율. `compute_metrics`와 early stopping 기준 eval에 사용됩니다.

`split.test_ratio`
: 최종 test 구간 비율

## 출력

Hydra output directory 아래에 다음이 저장됩니다.

- checkpoint
- `test_results.csv`
- `demand_error_analysis.png`
- `predictions_max_demand_node.png`
- `predictions_min_demand_node.png`
- `predictions_mid_demand_node.png`

## 문서

- 개요: [readme.md](/home/jinsu/PycharmProjects/DMVST/readme.md)
- 상세 파이프라인: [docs/pipeline.md](/home/jinsu/PycharmProjects/DMVST/docs/pipeline.md)
