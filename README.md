# MMDL — Qwen3-VL MMMU Evaluation

기존 서버의 `LLM.chat()` 스모크 테스트 방식을 사용합니다. 환경 설치, 드라이버 변경,
파일 삭제, 학습은 하지 않습니다. 실제 GPU 실행은 사용자 서버에서 검증해야 합니다.

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
  --output reports/ab_pixels.md
```

강의 슬라이드 해상도가 OOM 또는 context-length 오류를 내면 중간 범위인
`--min-pixels 262144 --max-pixels 1310720`으로 새 출력 디렉터리에서 실행합니다.
실패한 실행과 다른 설정의 결과를 합치지 않습니다.

해상도를 하나 고른 다음 프롬프트만 비교합니다. 두 실행 모두 출력 상한을 1024로 맞춰
프롬프트 외 조건을 동일하게 유지합니다.

```bash
bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU \
  --limit-per-subject 2 --prompt-style direct --max-tokens 1024 \
  --min-pixels 262144 --max-pixels 1310720 --batch-size 1 \
  --output-dir results/ab60_prompt_direct

bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct --data-root MMMU/MMMU \
  --limit-per-subject 2 --prompt-style cot --max-tokens 1024 \
  --min-pixels 262144 --max-pixels 1310720 --batch-size 1 \
  --output-dir results/ab60_prompt_cot

python scripts/compare_runs.py \
  results/ab60_prompt_direct results/ab60_prompt_cot \
  --label-a direct --label-b cot \
  --output reports/ab_prompt.md
```

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

### 900문항 전체 평가

소규모 실행 결과로 설정을 확정한 다음 실행합니다. 변경한 옵션은 전체 실행에도 똑같이 넣습니다.

```bash
bash scripts/run_mmmu_eval.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root MMMU/MMMU \
  --output-dir results/baseline900
```

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
- 시작값: batch 2, 전체 문맥 8192 tokens, 출력 최대 256 tokens,
  이미지당 min_pixels 65,536 / max_pixels 589,824, 최대 이미지 7장,
  GPU memory utilization 0.85. 이는 작은 초기 실행을 위한 공학적 선택이며 최적 설정이 아닙니다.
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

- `manifest.json`: revision/실행 인자/환경/소스 hash/데이터 fingerprint/성공·실패 상태
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
중단합니다.

```bash
python scripts/prepare_submission.py results/baseline900_final --name baseline900_final
```

이 명령은 최신 Gist 경로인 `reports/mmmu_baseline.md`, 실험 증거 파일을 담은
`artifacts/baseline900_final/`, 이전 PDF 경로와의 호환 링크인
`assignment/assignment1.md`를 만듭니다. 생성 후 TODO와 gap analysis를 사람이 완성해야 합니다.
환경 파일과 평가 코드를 함께 포함하고, 실행 커맨드의 경로와 옵션을 최종 설정에 맞춥니다.
주관식과 객관식, 이미지 참조 및 미파싱 샘플을 사람이 검토하세요.
성능 개선이나 GPU 실행 성공을 사전 검증했다고 주장하지 않습니다.
