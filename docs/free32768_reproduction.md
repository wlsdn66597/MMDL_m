# Free 32768: Qwen 공식 설정을 참고한 전체 재평가

## 목적과 해석

기존 free의 출력 상한 8192에서 남았던 잘림을 더 큰 예산으로 측정한다.
Qwen3-VL 공개 MMMU Instruct 실행 스크립트의 상한인 **32768**을 사용한다.
토큰을 무조건 끝까지 생성시키지 않는다. EOS가 나오면 즉시 종료한다.
상한 확대가 반복 생성이나 답 미완성을 반드시 해결하는 것은 아니다.

이번에는 프롬프트도 공식 자료에 맞춘다. 따라서 이전 8192 결과와의 차이는
**토큰 수만 바꾼 ablation이 아니라 공식 설정을 참고한 프로토콜 변경**이다.
기존 free/two-stage 결과는 그대로 보관한다. 이 실행에는 풀이 후 답을 다시 고르는
두 번째 모델 호출, 선택지 제한 디코딩, 정답을 이용한 재시도, 무작위 답 보완이 없다.

Qwen 보고서 Table 4의 4B-Instruct MMMU 수치는 67.4이다. 이를 참고 수치로 표시하되,
아래의 데이터 표현·전처리·채점 차이가 있으므로 동일 수치 재현을 보장하지 않는다.
보고서의 MMMU-Pro 53.2를 standard-4, standard-10, vision 각각의 목표값으로 해석하지 않는다.

## 출처

- [Qwen3-VL Technical Report, Table 4 및 Appendix B.1](https://arxiv.org/pdf/2511.21631)
- [공식 MMMU Instruct 실행 설정](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh)
- [공식 MMMU 프롬프트·추론 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)
- [공식 답 추출 및 judge 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/eval_utils.py)
- [공식 데이터·주관식 채점 전처리](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/dataset_utils.py)
- [MMMU 논문](https://arxiv.org/abs/2311.16502)
- [과제 공지](https://gist.github.com/neur-lab/38deabdfcde9e6dbacf362ab8059eb41)

코드 출처는 `96588727e44c78b25ba03ea03b8e12f7e64fd0da`로 고정했다.
확인한 Qwen 공개 evaluation 디렉터리에는 별도 MMMU-Pro 실행기가 없으므로,
Pro 프롬프트는 보고서 Appendix B.1을 참고한 구현이다.

## 고정 설정

설정 파일: `configs/free32768_reference_v1.json`.

| 항목 | 설정 |
|---|---|
| 모델 | Qwen/Qwen3-VL-4B-Instruct, BF16, 비양자화 |
| 모델 revision | ebb281ec70b05090aa6165b016eac8ec08e71b17 |
| 최대 출력 | 32768, 문항당 자유 생성 1회 |
| 전체 컨텍스트 | 65536: 입력과 최대 출력을 함께 수용 |
| 샘플링 | temperature 0.7 / top_p 0.8 / top_k 20 |
| 패널티 | repetition 1.0 / presence 1.5 |
| 시드 | 기존 실험의 3407 고정; 공개 MMMU 실행 코드의 engine seed 42와 다름 |
| 픽셀 제한 | min=1280×28×28, max=5120×28×28 |
| 실행 | 문항별 순차 처리, max_num_seqs=1 |
| 메모리 관련 | gpu_memory_utilization=0.85, eager, chunked prefill, prefill token batch=2048 |
| KV cache | auto: 양자화 KV 캐시를 강제하지 않음 |

공식 예시의 기본 컨텍스트는 128000이다. 여기서는 RTX 4090을 고려해 65536을 쓴다.
모든 문항에서 **실제 입력 토큰 + 32768 ≤ 65536**을 검사하며,
넘으면 입력을 자르거나 출력 상한을 몰래 줄이지 않고 실패한다.
CPU processor 검사와 vLLM의 실제 prompt token 수를 모두 기록한다.
출력 종료 원인이 length인데 생성 토큰 수가 32768보다 작아도 실패한다.

RTX 4090에서 이 새 설정의 실제 GPU 추론은 아직 검증하지 않았다.
OOM이면 로그를 보존하고 실행 설정을 검토해야 한다. 자동으로 해상도나 BF16을 바꾸지 않는다.
평균 출력이 짧아도 일부 반복 문항이 최대 32768까지 진행하므로 전체 실행은 오래 걸릴 수 있다.

## 문항 범위

| 실행 폴더 | 데이터 | 전체 문항 |
|---|---|---:|
| mmmu_val | MMMU validation, 30과목 × 30 | 900 = 객관식 847 + 주관식 53 |
| standard-4 | MMMU-Pro standard (4 options) test | 1730 |
| standard-10 | MMMU-Pro standard (10 options) test | 1730 |
| vision | MMMU-Pro vision test | 1730 |

MMMU revision은 `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`,
MMMU-Pro는 `563f3e84bb3b90893083a1f039cfa13077f2302b`이다.
세 Pro 설정의 ID 집합이 같은지 최종 요약 시 검사한다.
설정 이름과 다른 실제 선택지 수(예: 5, 9, 12개)도 모두 포함한다.
파싱 실패·잘림·오답 문항을 분모에서 제거하지 않는다.
최종 요약은 raw response 재채점, ID·중복·유형·과목별 범위와 단일 호출을 재검사한다.

MMMU는 과목 macro accuracy를 제출 기준으로 보고한다. 30×30에서는 micro와 같다.
Pro 주 지표는 전체 문항 micro accuracy이며 macro도 별도 기록한다.
기존 Pro test 실험들이 평가 정책 결정에 사용된 이력을 보고서에 밝히고,
개별 test 오답에 맞춘 추가 학습·하이퍼파라미터 선택은 하지 않는다.

## 공식 평가와 일치시키는 부분 / 남는 차이

MMMU는 이미지들을 먼저 배치하고 Question / Options / 정답 선택 요청을 붙인다.
기존의 정답 문자만 출력하라는 추가 지시는 제거했다. 주관식은 질문만 주고 자유 생성한다.
Pro standard는 질문·선택지·정답 선택 요청, vision은 스크린샷과 보고서의 step-by-step 요청을 쓴다.
vision 생성 입력에 데이터의 별도 질문 텍스트나 선택지 텍스트를 추가하지 않는다.

공식 MMMU 예시는 VLMEval TSV의 DEV_VAL을 로드한다. 과제 대상은 validation 900이므로
여기서는 기존 HF snapshot의 validation만 유지한다. dev 문항을 합쳐 점수를 내지 않는다.
공식 코드는 `qwen_vl_utils.process_vision_info`로 이미지 전처리 후 `LLM.generate`를 호출한다.
여기는 기존 HF 이미지의 투명도를 흰색 배경에 합성해 PNG로 직렬화하고,
vLLM `LLM.chat`의 HF processor에 동일 픽셀 제한을 준다. 이미지 표현·전처리 및
프레임워크 버전 차이까지 완전히 같다고 주장하지 않는다.
시드, 문항별 실행, 하드웨어, 컨텍스트 길이 차이도 결과 해석에 포함한다.

공식 Qwen 코드의 “two-stage evaluation”은 **규칙 기반 답 추출 → 별도 judge 추출**이다.
우리의 **VLM 풀이 → VLM 답 선택** two-stage와 다르다.
기본 GPU 실행은 기존 결정적 MC parser v2와 MMMU 공식 open parser로 채점하며,
원문·입력 fingerprint·실제 토큰·종료 원인·시간·패키지·템플릿을 보존한다.
추가 judge 채점은 아래 별도 명령으로 실행한다. 기존 점수를 덮어쓰지 않는다.

## 전체 실행

실험 저장소는 `wlsdn66597/MMDL_m`이다. 새 제출용 private `MMDL`과 구분한다.
서버 폴더 이름은 기존 `~/mmdl/MMDL`을 그대로 써도 된다.
첫 배포의 `free32768_reference_v1`은 입력 길이 검사 중 HF processor가 채팅 메시지의
이미지 유형을 바꾸면서 vLLM이 `Unsupported chat content part type: 'image'`로 중단될 수 있다.
수정본은 길이 검사용 메시지를 별도 복사해 원본을 보존한다. 코드 해시를 고정하는 재개
계약 때문에 v1 폴더를 수정본으로 이어 쓰지 않고 아래의 새 v2 폴더를 사용한다.
이전 폴더의 결과는 삭제하지 않는다. 기존 완료 건수는 다음과 같이 확인한다.

```bash
wc -l results/free32768_reference_v1/mmmu_val/predictions.jsonl 2>/dev/null || echo 'saved predictions: 0'
```

```bash
cd ~/mmdl/MMDL
git remote set-url origin https://github.com/wlsdn66597/MMDL_m.git
git pull --ff-only
source ../.venv-mmdl/bin/activate
mkdir -p logs
nohup bash scripts/run_free32768_reference.sh \
  results/free32768_reference_v2 \
  > logs/free32768_reference_v2.nohup.log 2>&1 < /dev/null &
echo $!
```

먼저 네 데이터 설정의 모든 입력을 확인한 뒤 GPU 실행을 네 번 순차 진행한다.
6090개의 새 응답을 생성한다. 기존 8192 응답을 이어 쓰거나 잘린 문항만 선택하지 않는다.
이 스크립트 자체에는 패키지 설치, 학습, judge API 호출이 없다.

```bash
tail -n 40 -f logs/free32768_reference_v2.nohup.log
# tail 화면은 Ctrl+C로 종료해도 실제 nohup 작업은 유지됨
cat results/free32768_reference_v2/status.txt
pgrep -af 'python.*(run_free_reproduction|eval_free_reproduction)\.py' || true
wc -l results/free32768_reference_v2/mmmu_val/predictions.jsonl 2>/dev/null || true
# complete 이후:
python scripts/summarize_free_reproduction.py results/free32768_reference_v2
cat results/free32768_reference_v2/summary.tsv
```

사전 입력 검사만 하려면 동일 실행기에 `--preflight-only`를 붙인다.
그 뒤 전체 실행을 시작하거나 중단된 실행을 재개할 때는 `--resume`을 붙인다.

```bash
nohup bash scripts/run_free32768_reference.sh \
  results/free32768_reference_v2 --resume \
  >> logs/free32768_reference_v2.nohup.log 2>&1 < /dev/null &
```

같은 root의 중복 실행은 파일 잠금으로 막는다. 완료 문항은 재검증 후 건너뛴다.
모델·데이터·코드·설정이 바뀌면 재개하지 않는다. 손상된 JSONL의 마지막 행도
임의로 삭제하지 않고 에러를 낸다. 중단 당시 완료 저장되지 않은 문항은 다시 생성한다.
추론 시간은 저장된 완료 응답의 호출 시간 합계, 전체 시간은 실패한 시도를 포함한 세션 합계다.
사전 입력 검사는 개별 inference run의 전체 시간에 포함되지 않는다.

## 선택: 원문을 그대로 두고 Qwen 방식으로 추가 채점

우선 API 없이 공식 규칙으로 추출 가능한 비율을 확인한다.

```bash
python scripts/rescore_free_qwen.py \
  results/free32768_reference_v1/mmmu_val \
  --output-dir reports/free32768_qwen_rules
```

`unresolved`가 있으면 `accuracy`는 null이다. `accuracy_lower_bound`는 미해결을
오답으로 포함한 하한이며 공식 점수가 아니다. judge에는 추론 원문과 질문·선택지를 보내므로
사용할 서비스와 모델을 명시해야 한다. 아래 값은 실제 이용 가능한 서비스로 지정한다.

```bash
# API key를 파일·Git에 저장하지 않고 환경변수로 설정한다.
read -rsp 'Judge API key: ' QWEN_JUDGE_API_KEY; echo
export QWEN_JUDGE_API_KEY
python scripts/rescore_free_qwen.py \
  results/free32768_reference_v1/mmmu_val \
  --output-dir reports/free32768_qwen_judge \
  --judge-model YOUR_JUDGE_MODEL \
  --api-base https://YOUR_API_HOST/v1
```

이 명령만 외부 judge API를 호출하며 사용료가 발생할 수 있다. GPU baseline 실행은 호출하지 않는다.
Qwen 코드의 기본 judge 표기는 `gpt-3.5-turbo-0125`이나 해당 모델·서비스의 실제 이용 가능성은
별도 확인해야 한다. 다른 judge를 쓰면 모델 이름을 기록하고 동일 공식 채점이라고 부르지 않는다.
judge 출력이 잘리거나 유효한 답이 아니면 미해결로 남긴다. API 오류는 중단되며
동일 명령에 `--resume`을 붙이면 저장된 채점부터 이어간다.

규칙·judge 프롬프트는 고정 Qwen 소스에서 가져왔다. 원본의 최대 25회 추출 재시도 및
마지막 무작위 선택은 재현하지 않는다. judge는 미해결 문항당 한 번만 호출한다.
주관식은 채점 단계에서만 `A=reference answer / B=Other Answers`로 바꾸며,
참조 정답을 VLM 생성에 전달하지 않는다. HF 정답 표현과 원본 TSV 표현 차이도 남는다.
이 도구는 공개 코드가 확인된 MMMU validation에만 적용한다.

`vendor/qwen_mmmu_extract.py`는 Qwen 코드에서 네 함수를 추출한 것으로,
출처 commit과 Apache-2.0 라이선스(`vendor/QWEN_LICENSE`)를 함께 보존했다.

## 로컬 검증 범위

CPU 단위 테스트와 모의 모델 통합 테스트로 900문항 완주·중단 재개·문항 포함·
단일 호출·상한 검사·채점 분리·원문 재검증을 확인한다. 실제 GPU 속도, 메모리 사용,
새로운 정확도와 32768에서도 남는 잘림 수는 위 서버 실행 결과로 확인해야 한다.
