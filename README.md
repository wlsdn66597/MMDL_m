# MMDL — Qwen3-VL MMMU Evaluation

기존 서버의 `LLM.chat()` 스모크 테스트 방식을 사용합니다. 환경 설치, 드라이버 변경,
파일 삭제, 학습은 하지 않습니다. 실제 GPU 실행은 사용자 서버에서 검증해야 합니다.

## 출력 정책 3조건 비교 (개발 실험)

### 현재 MMMU-Pro 전체 결과와 비교

파싱·종료 실패에 대한 최종 ablation은 이미 생성한 parser-v2 8192-token 자유 생성 결과를
그대로 사용합니다. `run_mmmu_pro_output_policy_ablation.sh`는 `free`를 다시 추론하지 않고,
각 설정의 1,730문항에 대해 `constrained`와 `two-stage`만 실행합니다. 총 6개 새 실행입니다.
기존 결과 폴더는 읽기만 하며 새 출력 루트에 보고서를 생성합니다.

```bash
cd ~/mmdl/MMDL
source ../.venv-mmdl/bin/activate
mkdir -p logs
nohup bash scripts/run_mmmu_pro_output_policy_ablation.sh \
  results/mmmu_pro_output_policy_full_v1 \
  results/mmmu_pro_tokens_parser_v2 \
  > logs/mmmu_pro_output_policy_full_v1.nohup.log 2>&1 < /dev/null &
echo $!
```

두 번째 인자는 parser-v2 결과의 루트이며 아래 세 디렉터리가 있어야 합니다.

- `tokens8192_standard-4`
- `tokens8192_standard-10`
- `tokens8192_vision`

시작 전에 기존 baseline이 1,730개 전체인지, 완료 상태인지, 평가 signature와 입력 파일이
있는지 검사합니다. 각 비교 단계에서는 ID, 정답, 문항 유형, 선택지 수, 프롬프트, 원본
이미지 해시, 모델 revision, dataset revision, sampling, 해상도, context가 일치하는지도
검사합니다. 기존 vision 로그에는 선택지 원문이 없으므로, pinned dataset revision과
선택지 수 및 이미지/프롬프트 해시로 입력 동일성을 확인합니다.

```bash
cat results/mmmu_pro_output_policy_full_v1/status.txt
tail -n 40 -f logs/mmmu_pro_output_policy_full_v1.nohup.log
# complete 이후:
cat results/mmmu_pro_output_policy_full_v1/summary.tsv
cat results/mmmu_pro_output_policy_full_v1/vision/comparison.md
```

이 실험은 MMMU-Pro test를 사용한 출력 정책 ablation으로 보고서에 공개합니다. 이 결과의
문항별 오류를 보고 프롬프트나 학습 방법을 다시 반복 조정하지 않습니다.

### 별도 소규모 개발 실험

`eval_output_policy.py`는 기존 평가 프로필과 분리된 실험 러너입니다. 기본 데이터는
MMMU validation의 **객관식만**이며, 각 과목에서 ID와 seed의 해시 순서로 4개씩 선택합니다.
정답/오답을 보고 선택하지 않습니다. 표본 ID와 이미지 수 분포를 저장하며, 세 조건의
ID·원래 메시지·이미지 해시·정답·샘플링·해상도·모델이 일치해야 비교 보고서를 생성합니다.
과목별 동일 개수 표본은 개발 진단용이며 900문항의 문항 유형 비율을 재현하지 않습니다.

| 모드 | 실행 | 기본 출력 예산 |
|---|---|---|
| `free` | 기존 direct 프롬프트로 자유 생성 | 8192 |
| `constrained` | 같은 direct 입력에 문항별 선택지 문자 제약 | 16 |
| `two-stage` | 짧은 풀이 후, 원본 이미지와 풀이를 보고 선택지 문자 제약 | 풀이 1024 + 최종 16 |

세 조건 모두 BF16, batch 1, context 16384, 동일 Qwen sampling recipe와 강의 해상도를
사용합니다. 선택지는 실제 개수에 맞춰 A부터 생성하며 12개 선택지도 지원합니다.
제약은 프롬프트 부탁이 아니라 vLLM `StructuredOutputsParams(choice=...)`로 적용합니다.
형식 제약이 어겨지면 실패하며 자유 생성으로 조용히 대체하지 않습니다.

두 단계에서는 원본 이미지가 최종 호출에도 유지됩니다. 풀이가 상한에 도달해도 최종
선택을 진행하되, 풀이 잘림과 최종 출력 잘림을 따로 기록합니다. 정답 라벨은 생성 호출에
전달하지 않습니다. `predictions.jsonl`의 `stages`에 두 응답·실제 샘플링·토큰·시간이 남습니다.
총 출력 토큰과 추론 시간은 두 호출의 합이며, 모델 로드 포함 시간도 별도로 보고합니다.
출력 예산이 의도적으로 다른 **방법/비용 비교**이며 동일 계산량의 ablation은 아닙니다.

```bash
cd ~/mmdl/MMDL
source ../.venv-mmdl/bin/activate
mkdir -p logs
nohup bash scripts/run_output_policy_ablation.sh results/output_policy_dev120_v1 \
  --per-subject 4 --max-tokens 8192 --reasoning-tokens 1024 --final-tokens 16 \
  > logs/output_policy_dev120_v1.nohup.log 2>&1 < /dev/null &
echo $!
```

소규모 명령은 세 모드를 한 GPU에서 순차 실행하며 모델은 모드마다 새로 로드합니다. 기존 출력 루트가
있으면 중단합니다. 실패/중단된 폴더를 자동 이어 붙이지 않으며 새 이름으로 실행합니다.
문맥 초과나 OOM은 기록 후 중단하고 입력/해상도/출력 예산을 자동 축소하지 않습니다.

```bash
cat results/output_policy_dev120_v1/status.txt
tail -n 40 -f logs/output_policy_dev120_v1.nohup.log
# complete 이후:
cat results/output_policy_dev120_v1/comparison.md
```

GPU 없이 입력 경로만 확인할 때는 아래처럼 별도 폴더에서 실행합니다.

```bash
python eval_output_policy.py --mode two-stage --check-only \
  --per-subject 4 --output-dir results/output_policy_input_check_v1
```

모드 하나만 실행할 때는 `--mode constrained` 또는 `--mode two-stage`를 사용합니다.
`--per-subject 0`은 MMMU validation 객관식 전체(847개)를 선택하고 주관식은 제외합니다.
현재 개발 실험에는 MMMU-Pro vision 화면 형태가 포함되지 않습니다. 그 형식까지
일반화한다고 결론 내리기 전에 별도 개발용 화면 이미지에서 검증해야 합니다.

최종 정책 고정 후 Pro 실험에도 같은 러너를 사용할 수 있습니다:
`--benchmark mmmu-pro --setting standard-4|standard-10|vision --per-subject 0`.
이는 명시적으로 요청한 test ablation으로 구분하며, 기본 스크립트는 Pro를 실행하지 않습니다.
과거 MMMU-Pro 점수와 이번 MMMU 개발 표본 점수를 직접 비교하지 않습니다. 새로운 정책을
최종 평가에 채택하면 base와 fine-tuned 모델 모두 동일 정책으로 비교해야 합니다.

새 러너의 제약 디코딩은 로컬 CPU 테스트로 호출/채점 흐름을 검증했습니다. 실제 설치된
vLLM의 제약 백엔드 및 4090에서의 GPU 동작은 서버 실행 결과로 확인해야 합니다.

## 시작

기존 가상환경은 그대로 사용하고, 코드만 별도 폴더에 clone합니다.

```bash
git clone https://github.com/wlsdn66597/MMDL.git ~/mmdl/MMDL
```

```bash
source ~/mmdl/.venv-mmdl/bin/activate
cd ~/mmdl/MMDL
```

환경 설정만 확인(설치·다운로드·추론 없음):

```bash
python scripts/check_env.py
```

각 패키지 버전, CUDA/GPU, 캐시 revision 및 모델 파일, 환경변수, 디스크 여유를 확인합니다.
문제 시 FAIL 항목을 먼저 해결하세요. 전체 서버 확인을 로컬에서 완료했다고 주장하지 않습니다.

문법/채점/입력 처리의 작은 테스트(추론하지 않음):

```bash
python -m unittest discover -s tests -v
```

### 먼저 60문항 확인

```bash
bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root MMMU/MMMU \
  --limit-per-subject 2 \
  --output-dir results/smoke60
```

지정 revision은 스크립트에 고정되어 있고, HF 캐시의 같은 파일은 재사용합니다.
모델은 해당 snapshot의 로컬 경로를 vLLM에 전달하므로 tokenizer/processor도 같은 snapshot을 사용합니다.
기존 smoke.py의 greedy, 64-token 출력, 최대 이미지 2장 설정을 그대로 사용하지 않습니다.

`[done] 60 questions`가 나온 뒤 아래 내용을 확인합니다.

```bash
cat results/smoke60/summary.json
```

파싱 실패, `length_limited`, peak VRAM을 보고 출력 예산/해상도의 근거를 정합니다.
각 과목 첫 두 문항으로 구성된 편의 표본이므로 60문항 정확도는 전체 성능 추정치로 쓰지 않습니다.
테스트에 주관식/다중 이미지 유형이 포함됐는지도 `inputs.jsonl`, `predictions.jsonl`에서 확인하세요.

문항별 생성·파싱 이상을 요약하려면 다음을 실행합니다.

```bash
python scripts/audit_predictions.py results/smoke60/predictions.jsonl --tails 500
```

### 통제된 A/B 실험

먼저 기존 해상도와 강의 슬라이드 해상도를 같은 60문항에서 비교합니다. 프롬프트와 나머지
설정은 동일하게 유지합니다. 높은 해상도는 메모리 사용량 때문에 batch 1부터 시작합니다.

```bash
bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU \
  --limit-per-subject 2 --prompt-style direct \
  --min-pixels 65536 --max-pixels 589824 --batch-size 1 \
  --output-dir results/ab60_pixels_low

bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU \
  --limit-per-subject 2 --prompt-style direct \
  --min-pixels 1003520 --max-pixels 4014080 --batch-size 1 \
  --output-dir results/ab60_pixels_course

python scripts/compare_runs.py \
  results/ab60_pixels_low results/ab60_pixels_course \
  --label-a pixels_low --label-b pixels_course \
  --allow-config-differences \
  --output reports/ab_pixels.md
```

강의 슬라이드 해상도가 OOM 또는 context-length 오류를 내면 중간 범위인
`--min-pixels 262144 --max-pixels 1310720`으로 새 출력 디렉터리에서 실행합니다.
실패한 실행과 다른 설정의 결과를 합치지 않습니다.

해상도를 하나 고른 다음 프롬프트만 비교합니다. 두 실행 모두 출력 상한을 512로 맞춰
프롬프트 외 조건을 동일하게 유지합니다.

```bash
bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU \
  --limit-per-subject 2 --prompt-style direct --max-tokens 512 \
  --min-pixels 262144 --max-pixels 1310720 --batch-size 1 \
  --output-dir results/ab60_prompt_direct

bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU \
  --limit-per-subject 2 --prompt-style cot-brief --max-tokens 512 \
  --min-pixels 262144 --max-pixels 1310720 --batch-size 1 \
  --output-dir results/ab60_prompt_cot_brief

python scripts/compare_runs.py \
  results/ab60_prompt_direct results/ab60_prompt_cot_brief \
  --label-a direct --label-b cot_brief \
  --allow-config-differences \
  --output reports/ab_prompt.md
```

`cot`은 제한 없는 단계별 풀이를 위한 진단 옵션입니다. 제출 후보 비교에는 세 단계 이내의
풀이와 명시적 종료를 요구하는 `cot-brief`를 사용합니다. `length_limited`가 발생하면 전체
900문항으로 확대하지 않습니다.

60문항은 설정 선택용입니다. 최종 점수는 선택한 설정으로 900문항을 새로 실행합니다.

기존 또는 최종 900 결과에서 주관식 오답 전체와 과목별 객관식 오답 2개를 뽑아 수동
오류 분석표를 만듭니다.

```bash
python scripts/make_error_audit.py \
  results/baseline900_final/predictions.jsonl \
  --inputs results/baseline900_final/inputs.jsonl \
  --mc-per-subject 2 --output reports/error_audit.csv
```

CSV의 `error_category`와 `notes`를 팀원이 직접 채웁니다. 권장 분류는
`perception/OCR`, `knowledge`, `reasoning/calculation`, `prompt/format`,
`parser/scoring`, `dataset/ambiguous`입니다.

입력만 900문항 확인하고 싶다면 다음 별도 실행을 사용할 수 있습니다(GPU 추론 없음).

```bash
python -u eval_mmmu.py --check-only \
  --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU \
  --output-dir results/input_check900
```

### 900문항 전체 평가: 고정 프로필

전체 해상도 실험에서 선택한 52.89% 조건을 `configs/mmmu_val_v1.json`에 고정했습니다.
direct prompt, 강의자료 해상도 범위, batch 1, 출력 256 tokens 및 Qwen sampling recipe를
검사한 뒤 실행합니다. 프로필과 충돌하는 옵션을 추가하면 추론 전에 중단됩니다.

```bash
bash scripts/run_mmmu_val_v1.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root MMMU/MMMU \
  --output-dir results/baseline900_v1
```

서버에 남아 있는 기존 52.89% 실행이 같은 조건인지 `manifest.json`으로 확인합니다.

```bash
python scripts/validate_run_profile.py results/ab900_pixels_course \
  --profile configs/mmmu_val_v1.json
```

새 프로필 실행은 `manifest.json`에 평가 signature를 기록하고 프로필 사본도 결과 폴더에
저장합니다. 이후 base 모델과 fine-tuned checkpoint를 비교할 때 `compare_runs.py`는 두
signature가 같은지 먼저 검사합니다. 해상도나 프롬프트처럼 설정 차이 자체가 실험 변수인
ablation만 `--allow-config-differences`를 명시합니다.

### 연구 지표 생성

학습 데이터를 정하기 전, 고정 베이스라인에서 정확도·출력 품질·시각 입력 구조별 성능을
한 번에 계산합니다.

```bash
python scripts/analyze_research_metrics.py results/baseline900_v1 \
  --output-prefix reports/baseline900_v1_metrics
```

다음 지표가 JSON과 Markdown으로 저장됩니다.

- 전체 정확도와 Wilson 95% 신뢰구간, 과목 macro accuracy
- 단일/다중 이미지, 명시적 단일/다중 이미지 참조별 정확도와 격차
- 미파싱, 다중 후보, 길이 제한, 기대 답변 형식 준수율
- 입력·출력 토큰 수, 중복 집계를 제거한 실제 batch 추론 시간과 처리량

base와 fine-tuned 모델처럼 같은 평가 설정의 두 실행을 비교하려면 다음과 같이 실행합니다.

```bash
python scripts/analyze_research_metrics.py results/baseline900_v1 \
  --paired-run results/finetuned900_v1 \
  --output-prefix reports/base_vs_finetuned_metrics
```

paired 결과에는 정확도 변화, 답변 변화율, correct→wrong/wrong→correct와 exact McNemar
검정이 포함됩니다. 프롬프트·해상도처럼 설정 차이 자체를 비교하는 ablation에만
`--allow-config-differences`를 추가합니다. 이미지 수와 참조 수에 따른 차이는 관찰 지표이며,
모델이 실제로 시각적 근거를 사용했다는 인과 증거는 아닙니다. 그 판단은 이후 이미지
제거·교체·순서 변경 같은 통제된 paired 실험으로 측정합니다.

### MMMU-Pro 3설정 베이스라인

#### 기존 출력의 파서 v2 재채점 (GPU 불필요)

문자 경계(`is`/`Cannot`의 첫 글자 오인), Markdown 강조, 줄바꿈 최종답 및 명시적
답변 거부 처리를 수정했습니다. 생성 조건은 그대로이며 파서 정책만 v2로 기록합니다.
기존 결과를 덮어쓰지 않고 별도 폴더에 새 predictions/summary/signature와 changes.json을
만듭니다. 실행 시간은 원래 추론 시간이며 재채점 시간은 rescore.json에 별도 저장합니다.

```bash
python scripts/rescore_mmmu_pro.py \
  results/mmmu_pro_tokens_v1/tokens4096_standard-4 \
  results/mmmu_pro_tokens_v1/tokens4096_standard-10 \
  results/mmmu_pro_tokens_v1/tokens4096_vision \
  results/mmmu_pro_tokens_v1/tokens8192_standard-4 \
  results/mmmu_pro_tokens_v1/tokens8192_standard-10 \
  results/mmmu_pro_tokens_v1/tokens8192_vision \
  --output-root results/mmmu_pro_tokens_parser_v2

cat results/mmmu_pro_tokens_parser_v2/rescore_summary.md
```

2048 결과도 같은 명령에 해당 실행 폴더들을 지정해 별도 output-root로 재채점할 수 있습니다.
비교 보고서를 다시 만들 때는 양쪽 모두 재채점된 폴더를 사용하세요.
기존 vision 로그에는 선택지 원문이 없어 기존 fallback 결과를 보존하면서 명시적 답변
파싱만 보수적으로 수리합니다. 표준 설정은 저장된 입력에서 선택지를 복원합니다.
이 보존 전략은 rescore.json에 기록됩니다. 새 실행은 선택지 원문도 저장합니다.
이 도구는 MMMU-Pro 전용이며 MMMU validation 재채점에는 사용하지 않습니다.

4096/8192 출력 예산 비교는 아래 별도 스크립트로 실행합니다. 두 예산 모두 문맥 길이
16384, batch 1 및 같은 프롬프트/해상도/샘플링을 사용합니다. 각 1730문항씩 총 6개 실행을
순차 처리하고 각 예산의 3설정 보고서와 예산 간 paired 비교를 자동 생성합니다.
기존 2048/9048 결과는 보존합니다. 기존 결과와의 비교에는 문맥 길이 차이도 있습니다.
MMMU-Pro에서 두 예산을 비교한 사실을 보고하고, 최종 설정 선택은 개발 데이터에서 합니다.

```bash
cd ~/mmdl/MMDL
source ../.venv-mmdl/bin/activate
mkdir -p logs
nohup bash scripts/run_mmmu_pro_token_sweep.sh results/mmmu_pro_tokens_v1 \
  > logs/mmmu_pro_tokens_v1.nohup.log 2>&1 < /dev/null &
echo $!
```

SSH 종료 후에도 실행됩니다. 다른 GPU 작업이 끝난 뒤 한 번만 시작하세요.
출력 루트가 이미 있으면 거절하며 기존 결과를 덮어쓰거나 자동 재개하지 않습니다.
오류 발생 시 그 단계에서 중단합니다. GPU 메모리 및 최대 이미지 입력과 8192 출력의
문맥 수용 여부는 서버에서 확인해야 하며, 오류 시 설정을 자동 축소하지 않습니다.

```bash
cat results/mmmu_pro_tokens_v1/status.txt
tail -n 40 -f logs/mmmu_pro_tokens_v1.nohup.log
# 완료 후:
cat results/mmmu_pro_tokens_v1/reports/generation_summary.tsv
```

상태가 `complete`이면 6개 실행과 보고서 생성까지 완료입니다. `failed:`이면
전체 로그 또는 출력 루트의 `logs/`에서 해당 실행 로그를 확인하세요.

MMMU-Pro는 test-only 1,730문항을 세 가지 형태로 제공합니다. 현재 데이터 revision과
평가 설정은 `configs/mmmu_pro_v1.json`에 고정했습니다.

이 프로필은 강의자료의 RTX 4090 권장값인 전체 문맥 9,048 tokens와 출력 최대
2,048 tokens를 사용합니다. Qwen upstream 재현 설정은 Instruct 모델에 더 긴 출력 상한을
사용하지만, 여기서는 과제 조건과 실행 비용을 맞추기 위해 2,048을 평가 상한으로 고정합니다.
상한까지 최종 선택지를 내지 못한 응답은 모델의 지시 이행 실패로 보고 오답 처리합니다.

- `standard-4`: 원래 4개 선택지
- `standard-10`: 같은 문항에 distractor를 늘린 10개 선택지
- `vision`: 질문과 선택지를 이미지 안에 넣은 vision-only 형식

설정 이름의 4/10은 데이터 구성 방식을 가리키며 모든 행의 선택지 수가 정확히 4/10이라는
제약은 아닙니다. 고정 공식 데이터에는 5개, 9개, 12개 선택지 문항도 있으므로 러너는 각 행의
실제 선택지 목록으로 A부터 필요한 문자까지 만들고 `option_count_distribution`을 결과에 기록합니다.

먼저 각 설정의 입력 1,730개를 GPU 없이 검사합니다. 서로 다른 출력 폴더를 사용합니다.
`scripts/check_env.py`에서 새 MMMU-Pro revision 캐시가 FAIL이면 아래 첫 명령이 해당 고정
revision을 받아오게 됩니다. 기존 Arrow 데이터가 재사용될 수 있지만 네트워크 연결은 필요합니다.

```bash
python -u eval_mmmu_pro.py --check-only \
  --setting standard-4 --output-dir results/mmmu_pro_check_standard4

python -u eval_mmmu_pro.py --check-only \
  --setting standard-10 --output-dir results/mmmu_pro_check_standard10

python -u eval_mmmu_pro.py --check-only \
  --setting vision --output-dir results/mmmu_pro_check_vision
```

전체 실행 전 Standard와 Vision 메시지 경로를 각각 소량 추론합니다. 이 결과는 성능 수치로
사용하지 않습니다.

```bash
python -u eval_mmmu_pro.py \
  --setting standard-10 --limit 10 \
  --output-dir results/mmmu_pro_smoke_standard10

python -u eval_mmmu_pro.py \
  --setting vision --limit 10 \
  --output-dir results/mmmu_pro_smoke_vision
```

두 실행에서 `unparsed`, `length_limited`, 입력 이미지 수와 원문 응답을 확인한 뒤 전체를
실행합니다. 일부 Qwen3-VL-4B-Instruct 응답은 direct 지시에도 2,048 tokens까지 풀이를
반복할 수 있습니다. 이 경우 토큰 상한을 계속 늘리거나 미완성 응답의 중간 선택지를
정답으로 복구하지 않습니다. smoke는 고정 제출 프로필이 아니므로 결과를 최종 점수와
합치지 않습니다.

고정 베이스라인은 세 번 실행합니다. 실행마다 모델을 새로 로드하므로 각각 별도 tmux에서
순차적으로 수행합니다.

```bash
bash scripts/run_mmmu_pro_v1.sh standard-4 results/mmmu_pro_standard4_v1
bash scripts/run_mmmu_pro_v1.sh standard-10 results/mmmu_pro_standard10_v1
bash scripts/run_mmmu_pro_v1.sh vision results/mmmu_pro_vision_v1
```

세 실행이 끝나면 동일 ID의 정오답 전이를 묶어 하나의 보고서를 생성합니다.

```bash
python scripts/summarize_mmmu_pro.py \
  --standard-4 results/mmmu_pro_standard4_v1 \
  --standard-10 results/mmmu_pro_standard10_v1 \
  --vision results/mmmu_pro_vision_v1 \
  --output-prefix reports/mmmu_pro_baseline
```

`standard-4→standard-10`은 추가 distractor 민감도, `standard-10→vision`은 질문과 선택지를
이미지에서 읽는 OCR·시각 입력 부담을 나타냅니다. 두 차이는 동일 ID의 paired accuracy와
exact McNemar 검정으로 보고합니다.

MMMU-Pro는 최종 일반화 지표이므로 이 실행을 **학습 전 베이스라인으로 동결**합니다.
개별 test 정답·오답을 보고 학습 데이터, epoch 또는 하이퍼파라미터를 고르지 않습니다.
학습 방향과 오류 유형은 MMMU validation 결과 및 별도 개발 데이터에서 결정하고,
MMMU-Pro는 학습 전 한 번과 최종 checkpoint 한 번만 비교하는 것을 원칙으로 합니다.

SSH 접속이 끊길 수 있으면 먼저 `tmux new -s mmmu_eval`을 실행하고, 그 안에서
위의 가상환경 활성화/폴더 이동/평가 명령을 실행하세요. Ctrl-B 다음 D로 분리합니다.
다시 접속할 때는 `tmux attach -t mmmu_eval`입니다.

출력 폴더가 이미 있으면 중단하도록 설계했습니다. 재실행에는 `results/smoke60_v2`처럼
새 이름을 사용하세요. 부분 결과를 다른 설정의 결과와 이어 붙이지 않습니다.
오류 문항을 건너뛰거나 임의로 문맥을 잘라서 정상 완료처럼 표시하지 않습니다.
에러가 나면 `logs/`의 로그와 해당 실행의 `manifest.json`을 확인하세요.

## 기본값과 선택 근거

- BF16, 양자화 없음. 모델 revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- MMMU revision: `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation만 평가.
- 30개 과목 전체에서 각각 30개인지, 문항 ID 900개가 유일한지 검증.
- Qwen Instruct 공식 평가 recipe: temperature 0.7, top_p 0.8, top_k 20,
  repetition_penalty 1.0, presence_penalty 1.5, seed 3407. 추론당 답변 1개.
- MMMU-Pro 고정 프로필은 강의자료의 RTX 4090 권장값에 맞춰 전체 문맥 9048 tokens와
  출력 최대 2048 tokens를 사용합니다. 선택지 제한 디코딩은 공식 direct prompt보다
  강한 제약이므로 베이스라인에 사용하지 않고, 필요하면 별도 ablation으로 보고합니다.
- 탐색 시작값은 batch 2, 이미지당 min_pixels 65,536 / max_pixels 589,824였습니다.
  고정 `mmmu_val_v1`은 900문항 실험 결과에 따라 batch 1, 전체 문맥 8192 tokens,
  출력 최대 256 tokens, min_pixels 1,003,520 / max_pixels 4,014,080,
  최대 이미지 7장, GPU memory utilization 0.85를 사용합니다.
- 이미지의 실제 가로/세로는 모델 processor가 종횡비와 patch 규칙에 맞춰 처리합니다.
  이미지를 항상 768×768 정사각형으로 만드는 설정은 아닙니다.
- 출력 상한은 `--max-tokens`, 문맥 길이는 `--max-model-len`, 해상도는
  `--min-pixels` / `--max-pixels`, 배치는 `--batch-size`로 명시적으로 바꿀 수 있습니다.
- `VLLM_USE_FLASHINFER_SAMPLER=0`은 이미 성공한 서버의 nvcc 부재 우회 설정을 유지합니다.
- 공식 MC 파서의 무작위 fallback을 없애고 미파싱을 오답 처리합니다. 주관식은
  고정 commit의 공식 MMMU 파서/채점 함수를 사용합니다. 자세한 내용은 THIRD_PARTY.md.
- MC 프롬프트는 선택지 문자 하나만, open 프롬프트는 간결한 최종 답만 요구합니다.
  길이 제한에 걸렸고 명시적 최종 답도 없는 MC 응답은 중간에 언급된 문자를 채점하지 않습니다.
- 정확히 같은 seed라도 라이브러리, GPU, 배치 구성을 바꾸면 출력이 달라질 수 있습니다.

## 데이터 경로와 체크포인트

`--data-root`에는 `MMMU/MMMU` 또는 지정 revision의 실제 HF snapshot 디렉터리를 줍니다.
`~/.cache/huggingface/hub` 자체는 데이터셋 snapshot이 아니므로 주지 않습니다.
datasets의 Arrow 캐시 위치를 바꾸려면 별도 `--cache-dir`을 사용합니다.

향후 파인튜닝 후에는 `--model-path /path/to/full_checkpoint`로 변경할 수 있습니다.
원격의 다른 체크포인트라면 `--model-revision`도 해당 revision으로 지정해야 합니다.
이 러너는 독립 HF 체크포인트를 받으며, LoRA adapter-only 폴더 로딩은 구현하지 않았습니다.
과제1 기준 모델이 아닌 체크포인트 실행은 보고서에 표시됩니다.

## 저장되는 결과

- `manifest.json`: revision/실행 인자/환경/소스 hash/평가 signature/데이터 fingerprint/성공·실패 상태
- `evaluation_profile.json`: 고정 프로필을 사용한 실행의 프로필 사본
- `predictions.jsonl`: 문항별 원문 응답, 추출 답, 정답, 파싱 모드, 토큰 수, 종료 사유
- `inputs.jsonl`: 정답·해설이 없는 실제 사용자 메시지(이미지 바이너리는 생략), 이미지 번호/크기/hash
- `chat_template.txt`: 모델의 실제 chat template
- `sampling_params.txt`: 실제 vLLM sampling 설정
- `summary.json`: 문항 유형, 6개 분야, 30과목 점수/시간, 반올림 전 macro·micro,
  미파싱/다중 후보/길이 제한 건수
- `requirements.freeze.txt`, `environment.txt`: 서버 실측 패키지 목록과 nvidia-smi
- `report_draft.md`: 제출 템플릿의 항목 순서를 따르는 초안. 60문항 실행은 개발용이라고 표시
- `logs/`: stdout/stderr 로그

시간은 `total_seconds`에 데이터 로딩과 모델 초기화를 포함합니다. 과목별 시간은 해당 과목의
입력 준비·추론·채점·저장 시간을 포함하며, 공통 모델 초기화는 제외합니다.
배치 추론 시간은 문항 각각의 독립 지연시간이 아닙니다. 같은 batch_id의 batch_seconds를
문항마다 더하면 중복 집계됩니다.

VRAM은 nvidia-smi의 물리 GPU 0 전체 사용량을 1초 간격으로 측정합니다.
vLLM 자식 프로세스뿐 아니라 다른 사용자의 프로세스도 포함하며 순간 최대치를 놓칠 수 있습니다.
CUDA_VISIBLE_DEVICES로 다른 GPU를 사용하면 `--monitor-gpu`에 그 물리 번호/UUID를 지정하세요.
모니터가 실패하면 값은 null로 남으며, 실제 값처럼 추정해서 채우지 않습니다.

## 제출

최종 900문항 실행의 `report_draft.md`를 팀 repo의 `reports/mmmu_baseline.md`로 옮긴 뒤,
팀 정보/환경 설명/설정 근거/1000자 이내 격차 분석을 직접 완성합니다.

완료된 실행을 제출 구조로 복사하려면 다음 명령을 사용합니다. 900문항이 아니면 스크립트가
중단합니다. 또한 현재 `mmmu_val_v1` 프로필 hash가 없는 이전 결과나 프로필 밖 설정의
결과도 제출용으로 복사하지 않습니다.

```bash
python scripts/prepare_submission.py results/baseline900_v1 --name baseline900_v1
```

이 명령은 최신 Gist 경로인 `reports/mmmu_baseline.md`, 실험 증거 파일을 담은
`artifacts/baseline900_v1/`, 이전 PDF 경로와의 호환 링크인
`assignment/assignment1.md`를 만듭니다. 생성 후 TODO와 gap analysis를 사람이 완성해야 합니다.
환경 파일과 평가 코드를 함께 포함하고, 실행 커맨드의 경로와 옵션을 최종 설정에 맞춥니다.
주관식과 객관식, 이미지 참조 및 미파싱 샘플을 사람이 검토하세요.
성능 개선이나 GPU 실행 성공을 사전 검증했다고 주장하지 않습니다.
