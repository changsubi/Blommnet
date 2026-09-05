# BloomNet

녹조 영역 segmentation + 픽셀별 Chl-a 회귀 (K-water T1, DGIST 담당).

- 설계 헌법: [`methods/spec/00_설계헌법.md`](../methods/spec/00_설계헌법.md) — **모든 코드가 따라야 하는 규약**
- API 동결표: [`methods/spec/06_매니페스트_API.md`](../methods/spec/06_매니페스트_API.md)
- 정정사항(06 과 충돌 시 **07 이 우선**): [`07_정정사항_A.md`](../methods/spec/07_정정사항_A.md) · [`07_정정사항_B.md`](../methods/spec/07_정정사항_B.md)

---

## 0. 레이아웃과 실행 위치

이 디렉터리(`k_water/bloomnet/`)가 곧 파이썬 패키지 `bloomnet` 이다.
**모든 명령은 부모 디렉터리에서 실행한다.**

```bash
cd <repo_root>          # ← 항상 여기서
```

```
k_water/
├── configs -> bloomnet/configs      심볼릭 링크 (06 §2.1 매니페스트의 저장소 루트 `configs/` 규약용.
│                                     config.py 의 기본 `--base` 경로 "configs/_base.yaml" 이
│                                     cwd=k_water 에서 그대로 풀리게 한다. 실체는 하나뿐이다)
└── bloomnet/                 ← 파이썬 패키지 (= 이 README 가 있는 곳)
    ├── configs/              설정 YAML (_base.yaml 이 스키마 정본)
    ├── tools/                실행 진입점 5종 (train / train_spec / eval / export / benchmark)
    ├── constants.py          모든 리터럴의 단일 출처
    ├── config.py             스키마 dataclass 트리 + YAML 로드/병합/검증
    ├── data/ modules/ models/ losses/ engine/ pretrain/ deploy/ utils/
    └── tests/
```

## 1. 설치

의존성은 3분할이다 (06 §5.0, 정정 A-13).

| 파일 | 용도 | 현 상태 |
|---|---|---|
| `requirements.txt` | 런타임 | pip 로 설치 |
| `requirements-dev.txt` | 테스트·정적검사 | `pytest` 설치됨 / `pytest-cov`,`ruff` 미설치 |
| `requirements-deploy.txt` | ONNX export | **전부 미설치** — export 게이트 미검증 |

```bash
cd <repo_root>
python -m bloomnet.tools.export --deps          # 설치 현황 출력 (조용한 skip 금지, 06 §5.0)
```

> 저장소 루트의 `requirements.txt` 로 환경을 만든다.
> 헌법 C-5.3 이 인터넷 의존을 금지하므로, `requirements-deploy.txt` 는 오프라인 wheel 확보가
> 선행 과제다 (06 §7.3 B5).

## 2. 데이터 준비

원본은 **복사하지 않는다** (52 GB). 심볼릭 링크 트리 + 군(flight-line) 분할을 만든다.

```bash
cd <repo_root>

# (1) aihub092 심볼릭 링크 트리                                    [01 §7.1]
python -m bloomnet.tools.link_aihub092 \
    --src <AIHUB092_ROOT> \
    --out data/aihub092_asis

# (2) 12클래스 픽셀/이미지 분포 (RFS 입력, U-9 해소)                  [05 §4.1]
python -m bloomnet.tools.compute_class_stats \
    --root data/aihub092_asis --split train --out data/cache/class_stats.json

# (3) flight-line 군 분할 — SAT "학습 미노출" 증빙               [01 §7.3, 정정 A-1]
python -m bloomnet.tools.make_group_split \
    --root data/aihub092_asis --out_root data/aihub092_group --scan_workers 16
#   ★ --presence_cache 를 주지 않는다. compute_class_stats 의 캐시는 train split(85,635행)만 담고
#     make_group_split 은 train+val+test(96,340장)를 스캔해 행 수 검사에 걸린다. 자체 병렬 스캔이 수십 초면 끝난다.
#   실측 기대값(SEED=20260731, 96,340장):
#     67,438 / 14,454 / 14,448 = 70.000 / 15.003 / 14.997 %, max_abs_dev 3.11e-5,
#     군 129/21/24, 위반 0건, exempt={id 8}, retries_used=0

# (4) 235 다중분광 칩 npz 캐시 (S0-Spec 용)                          [01 §7.6]
python -m bloomnet.tools.build_k235_cache --out data/cache/k235_ms.npz
```

> **위 4개 모두 구현 완료**되어 `--help` 가 동작한다(통합 단계에서 채움). `data.sampler.class_stats`
> 를 `null` 로 두면 데이터셋이 라벨을 전수 스캔하므로(느림) (2)를 먼저 실행하는 편이 좋다.

> **왜 기본 분할이 `group` 인가:** 01 §7.3 [M3] 이 기존 asis 분할의 누수를 실측했다.
> `configs/_base.yaml` 의 `data.root` 는 `data/aihub092_group` 이 기본이고,
> asis 비교는 `--set data.root=data/aihub092_asis --set data.split_variant=asis` 로 한다.

## 3. 학습

### 3.1 S0-RGB — AI Hub 092 도메인 근접 사전학습 (오늘 실행 가능)

```bash
cd <repo_root>
CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.train \
    --config bloomnet/configs/s0_rgb_aihub092.yaml \
    --set data.sampler.class_stats=data/cache/class_stats.json
```

- `--config` 는 경로 대신 **프리셋 이름**으로도 된다: `--config s0_rgb_aihub092`
- 결과는 `outputs/<run_name>/` 에 쌓인다. `run_name` 미지정 시 `{mode}_{git_rev}_{timestamp}`.
- `run_dir/config.resolved.yaml` 에 **병합 완료본 전체**가 덤프된다 (재현성, 06 §4.1 step 6).

> ⚠️ `schedule.batch_size: 32` 는 **추정치**다. 첫 실행 전 05 §5.1.8 절차(모델+criterion
> end-to-end 로 B ∈ {8,16,24,32,48} 실측)를 반드시 수행한다. `warmup_iters`/`total_iters` 는
> 절대값이 아니라 `warmup_epochs`·`epochs` 에서 런타임 유도되므로 B 를 바꿔도 따라간다
> (정정 A-18 / B-26).

### 3.2 S0-Spec — 235 칩 Chl-a 회귀 (CPU, 오늘 실행 가능)

```bash
cd <repo_root>
CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.train_spec \
    --config bloomnet/configs/s0_spec_k235.yaml
```

- `SpecMLP` 는 1,089 params 라 GPU 가 필요 없다.
- 산출물: `outputs/<run>/spec_gate.json` — LOSO 8-fold 지표 + 이식 게이트 G1~G3 판정.
- **이 스크립트는 이식(transplant)을 수행하지 않는다** (정정 A-19). 밴드 순서가 외부 문서로
  확정(`spec.band_order_confirmed=true`)되기 전까지 V18 이 이식을 거부한다.
  게이트 실패는 "분광 단독으로는 안 된다"는 **정량 근거**이며 기대 결과에 포함된다 (01 §6.9).

### 3.3 S1 / S2 — 라벨·센서 대기

```bash
CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.train \
    --config bloomnet/configs/s1_rgb_ms4.yaml \
    --init_from outputs/<s0_rgb_run>/best.pt
```

두 config 모두 **오늘 실행 불가**이며 선결 조건이 파일 상단 주석에 적혀 있다:
유니바 픽셀 라벨(U-3), 그리고 `tools/calibrate_k_sensor.py` 로 `K_SENSOR["m3m"]` 재산출(M-13).

### 3.4 자주 쓰는 오버라이드

```bash
# 중첩 키는 --set (반복 가능, 값은 yaml.safe_load 로 파싱)
--set model.ela.head_dim="[16,16,32,32]"  --set train.ema.enabled=false

# 1급 플래그 15종은 --set 으로 자동 변환된다
--batch_size 8 --lr 6e-4 --epochs 80 --seed 7 --device cuda --num_workers 8
--amp bf16 | --no_amp   --resume_from <ckpt>   --init_from <ckpt>   --num_classes 2
--run_name my_run --output_dir outputs --data_root data/aihub092_asis --dry_run

# store_true 를 YAML/CLI 에서 **끌 수도** 있다 (이전 구현의 결함 제거)
--set train.ema.enabled=false
```

### 3.5 Ablation

```bash
python -m bloomnet.tools.train --config a1_dice              # w_dice 0.4 → 0.0
python -m bloomnet.tools.train --config a2_rfs               # RFS t 0.05 → 0.0
python -m bloomnet.tools.train --config a3_layerscale        # γ init 0.01 → 0.1 (stage1~2)
python -m bloomnet.tools.train --config a4_bio_gndvi         # x_bio 2번째 채널 MCI_norm → GNDVI
python -m bloomnet.tools.train --config a5_aux_p8            # aux tap 인코더 3개 → 디코더 p8
python -m bloomnet.tools.train --config a6_attn_all_stages   # LiteMLA stage 3,4 → 1,2,3,4
```

## 4. 평가

```bash
cd <repo_root>
CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.eval \
    --config bloomnet/configs/s0_rgb_aihub092.yaml \
    --ckpt outputs/<run>/best.pt --split test --ema \
    --out outputs/<run>/eval_test.json
```

- mIoU 규약은 `exclude_absent` **고정**이다 (union==0 클래스 제외, X-20 / 헌법 C-12).
- 부트스트랩 95% CI 는 이미지가 아니라 **flight-line 군 단위** 리샘플이다 (정정 A-4).
- `f1_at_100` 은 235 에 ≥100 mg/m³ 표본이 0개라 값 대신 `null` + 사유가 기록된다 (정정 A-30).

## 5. 배포 (ONNX / TensorRT)

```bash
cd <repo_root>

# onnx 없이도 계약만 먼저 검사 (deploy() 가지치기 + ExportWrapper 3-tuple + conf/chl 범위)
CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.export \
    --config s1_rgb_ms4 --check_only --device cpu --input_hw 64 64

# 실제 ONNX 생성 (requirements-deploy.txt 필요)
CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.export \
    --config bloomnet/configs/s1_rgb_ms4.yaml \
    --ckpt outputs/<run>/best.pt --out outputs/<run>/bloomnet.onnx

# 04 §9.4 step 4 — ORT vs TRT 대조 (수동)
polygraphy run outputs/<run>/bloomnet.onnx --trt --fp16 --onnxrt --atol 1e-2 --rtol 1e-2
```

- `deploy()` 후 `state_dict` 에 `edge_head|aux_|siam` 키가 **0개**여야 한다 (04 §9.4 step 1).
- `Shape`/`Gather` 노드 0개 검사는 **ExportWrapper 산출 ONNX 에만** 적용한다 (정정 A-21).
  학습 그래프의 `F.interpolate(size=…)` 는 규칙 위반이 아니다.
- `conf ∈ [0.029312, 0.970688]` — `s ∈ [−7,7]` hard clamp 때문에 1 에 도달하지 않는다 (X-25/X-26).

## 6. 벤치마크

```bash
cd <repo_root>

# 파라미터 + MAC 만 (CPU 로 충분. 정정 A-24: 256² 실측 후 ×4 / ×16 스케일)
CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.benchmark \
    --config s2_full --device cpu --macs_only --deploy

# latency (이전 구현 프로토콜 복제: untimed 1 → warmup 20 → sync → timed 100 → sync)
CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.benchmark \
    --config bloomnet/configs/s1_rgb_ms4.yaml \
    --input_hw 1024 1024 --bench_batch 1 --iters 100 --warmup 20 --deploy
```

- `--deploy` 를 주면 학습 전용 헤드(edge/aux/siam)를 제거한 **추론 구성**으로 측정한다.
  06 §6.2 의 예산표(S2-Full K=2 = 15,969,023 params / 62.678 GMAC@1024²)와 대조하려면 필수다.
- CPU 로 돌리면 iters/warmup 이 자동 축소되고, 그 latency 는 Jetson Orin 실시간성 판단의
  근거가 될 수 없다는 경고가 출력된다.

## 7. 테스트

```bash
cd <repo_root>
CUDA_VISIBLE_DEVICES="" python -m pytest bloomnet/tests -q -m "not data and not slow"  # 필수 게이트
CUDA_VISIBLE_DEVICES="" python -m pytest bloomnet/tests -q -m data                     # 실데이터 접근
CUDA_VISIBLE_DEVICES="" python -m pytest bloomnet/tests -q -m slow                     # onnx export
```

**GPU 를 절대 쓰지 않는다** (헌법 C-5.2 — 별도 학습이 GPU 를 점유 중). 모든 테스트는
`CUDA_VISIBLE_DEVICES=""` 로 실행하며, autouse fixture 가 `torch.cuda.is_available() == False`
를 assert 한다.

## 8. 설정 스키마

`bloomnet/configs/_base.yaml` 이 스키마의 **정본**이며, dataclass 트리의 덤프본이다.
여기 없는 키는 로드 시 전체 dotted path 와 함께 `ValueError` 로 거부된다.

```bash
# dataclass 기본값에서 재생성 (손으로 고치지 않는다)
cd <repo_root>
python -m bloomnet.config --dump-base --out bloomnet/configs/_base.yaml

# 병합·검증 결과만 확인 (모델·데이터 생성 없음)
python -m bloomnet.tools.train --config s1_rgb_ms4 --print_config
```

우선순위는 **CLI `--set` > `--config` (의 `_base:` 체인) > `configs/_base.yaml` > dataclass 기본값**
이며, 리스트는 병합이 아니라 **교체**된다 (06 §4.1).

검증 규칙 V1~V22 는 `config.py::_validate_rules` 에 있다. 자주 걸리는 것:

| 규칙 | 내용 |
|---|---|
| V2 | `lambda_seg=1.0`, `lambda_reg=0.5` 는 **사업문서 확정값**. `--allow_contract_break` 로만 우회 |
| V5/V16 | `bio` 는 `msi` 의 부속이다. `kind ∈ {mci,gndvi} ⟹ source == msi` |
| V6 | `model.ppn.use_pol` 은 `'pol' in modalities` 에서 자동 동기화. 명시 불일치는 에러 |
| V12/V18 | 이식은 `bio.source==msi ∧ kind==mci ∧ band_order_confirmed` 일 때만 |
| V19 | `1 ∈ model.bmef.stages` 필수. stage1 제외 ablation 은 `bmef.stage1_identity: true` |
| V21 | `sensor != none ⟹ msi.k_sensor > 0` |

## 9. 절대 하지 말 것 (06 §10 요약 카드)

1. GPU 사용 (헌법 C-5.2)
2. `msi`/`bio`/`ir`/`pol` 에 광도 증강
3. 라벨을 nearest 로 다운샘플 (로짓을 라벨 해상도로 올린다)
4. BMEF 융합부를 fp16 으로 실행 (`w_min = 3.07e-6` 은 fp16 subnormal)
5. `(B,3,C,H,W)` softmax 텐서 materialize
6. `exp(A−amax)` 에서 `clamp(max=0)` 생략
7. `AdaptiveAvgPool2d(1)` 사용 (→ `x.mean`)
8. **export 그래프에서** `F.interpolate(size=…)` (학습 경로에서는 `size=` 가 정본)
9. `bio` 를 기하변환 **전에** 계산
10. 편광 flip 시 AoLP 변환 생략
11. mIoU 를 `union==0` 포함으로 계산
12. `best_epoch_*.pt` 누적 저장 (`best.pt`/`last.pt` 2개만)
13. `torch.expm1` 사용 (ONNX 미지원 → `exp(u)−1`)
14. `msi` 를 R4′(`k_sensor`) 없이 학습에 투입
15. hot path 에서 `.any()`/`.max()` 파이썬 bool 변환
