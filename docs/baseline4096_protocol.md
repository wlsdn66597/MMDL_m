# Two-stage 4096 baseline: requirement audit and frozen protocol

검토일: 2026-09-23. 실제 GPU 실행 결과를 기록한 문서가 아니라 이번 코드의 실행 계약입니다.

## 근거와 필수 범위

- [과제 공지](https://gist.github.com/neur-lab/38deabdfcde9e6dbacf362ab8059eb41)
- [제출 템플릿](https://gist.github.com/neur-lab/483852e1f9d8d52f54627e600677c2f9)
- 강의자료 `02_MMDL_Team_Project_Introduction.pdf` 및 RTX 4090 권장 설정 슬라이드
- [Qwen 공식 평가 설정](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction)
- [MMMU-Pro 공식 데이터](https://huggingface.co/datasets/MMMU/MMMU_Pro)

| 항목 | 과제/자료에서 요구하거나 안내한 내용 | 이번 구현 |
|---|---|---|
| 과제 1 모델 | Qwen3-VL-4B-Instruct 원본, BF16, 비양자화 | 기존 공식 revision 고정, 양자화 checkpoint 거부 |
| 과제 1 데이터 | MMMU validation 900 전체, 30과목 | 과목당 30, ID 중복 없음, MC 847 + open 53 모두 검증 |
| 메인 지표 | 30과목 accuracy 평균 | macro 및 micro 모두 저장; 30×30에서는 동일 |
| 프롬프트/생성/채점 | 실제 설정·근거를 명시하고 재현 가능하게 고정 | profile, 입력/이미지 해시, 두 단계 출력, scorer revision 기록 |
| backend | vLLM 사용 가능 | 기존 vLLM chat 경로 유지, 한 GPU에서 순차 처리 |
| 경로 | 모델과 데이터 경로 CLI 입력 | 모델/데이터 경로 override 제공 |
| 제출물 | 과목별/전체 결과, 환경, VRAM, 시간, 설정, gap analysis | 보고서 초안과 artifact 복사 지원; 학생 정보와 gap 해석은 작성 필요 |
| 참조 점수 | 67.4%, 차이에 대한 근거 있는 분석 | 차이 자동 계산; 동일 재현이라고 주장하지 않음 |
| 팀 프로젝트 | fine-tuning 전후 MMMU/MMMU-Pro 및 ablation | 이번에는 사전 baseline만 생성; 학습과 사후 비교는 다음 단계 |

최신 공지의 제출 보고서 경로는 `reports/mmmu_baseline.md`입니다.
이전 강의자료의 `assignment/assignment1.md`에는 위 보고서 링크를 생성합니다.
기존 `prepare_submission.py`가 free 전용 profile만 허용하던 부분도 수정했습니다.

MMMU-Pro는 세 설정 각각 1,730개입니다. 이름의 4/10이 모든 행의 실제 선택지 수를 보장하지 않습니다.
5/9/12개인 행도 버리지 않고 각 행의 실제 2–26개 선택지를 그대로 사용합니다.
vision은 질문/선택지가 포함된 이미지를 입력하며 정답이나 별도의 질문/선택지 텍스트를 누출하지 않습니다.

## 이번 baseline이 의미하는 것

공지에는 한 번의 자유 생성만 사용하라는 요구가 없고 프롬프트/생성 설정의 문서화를 요구합니다.
따라서 two-stage를 **우리의 고정 평가 프로토콜**로 명확히 보고하고 학습 전후 동일하게 적용합니다.
이는 공식 점수 67.4%의 완전한 재현이나 강의가 특별히 승인한 two-stage 방법이라는 뜻은 아닙니다.
기존 free 결과는 별도 비교 기준으로 보존합니다.

Qwen 공식 Instruct 설정의 temperature=.7, top_p=.8, top_k=20, repetition_penalty=1,
presence_penalty=1.5, seed=3407을 두 단계 모두에 적용합니다.
공식 `out_seq_length=32768`과 달리 **우리의 풀이 예산은 4096**입니다.
강의 슬라이드의 2048도 권장 설정이며, 4096/two-stage는 계산 비용과 완료율을 함께 보고하는 자체 프로토콜입니다.
최종 객관식의 제한 디코딩은 확률분포를 변경하므로 sampling 값이 같다고 자유 생성과 동일한 조건인 것은 아닙니다.

| 설정 | 고정값 |
|---|---|
| 풀이 단계 | 최대 4096토큰, 짧은 시각적 근거/필요한 계산을 요청 |
| MC 최종 단계 | 원본 입력+풀이를 다시 참고, 실제 선택지 문자 중 하나, 최대 16토큰 |
| Open 최종 단계 | 원본 입력+풀이를 다시 참고, 단위 포함 짧은 답, 최대 128토큰; 문자 제약 없음 |
| Context | 16384; 이미지/질문/풀이/최종 지시와 답변까지 포함 |
| 이미지 크기 범위 | 1003520–4014080 pixels (1280×28²–5120×28²) |
| batch / GPU memory fraction | 1 / .85 |
| 이미지 참조/배치 | 기존 번호 부여 및 원본 순서 유지; 학습 전 baseline에서 재배열하지 않음 |

두 번째 단계에는 첫 단계 입력의 원본 이미지를 그대로 유지하며 정답 레이블은 전달하지 않습니다.
풀이가 잘려도 두 번째 단계를 실행합니다. `reasoning_length_limited`가 그 경우를 별도로 셉니다.
`unparsed=0`은 답 형식의 유효성을 뜻하며 풀이 완결이나 정답을 보장하지 않습니다.
주관식은 **최종 답변만** vendored MMMU open parser/evaluator로 채점합니다.
빈 답이나 최종 단계 길이 초과는 미파싱/오답으로 처리하고 분모에서 제외하지 않습니다.
나중에 다른 채점 규칙을 쓰려면 같은 원문 출력에 대해 양쪽 모델을 같은 규칙으로 재채점해야 합니다.

## 완전성 검사와 보관 파일

1. GPU 로딩 전에 네 데이터셋 입력 검사를 모두 수행합니다: 900 + 1730×3 = 6090 평가 레코드.
2. MMMU의 과목/유형/ID, Pro의 각 전체 ID 수, 이미지 참조, 빈 이미지, 정답 선택지 범위를 검사합니다.
3. 추론 시작 때 preflight와 설정 signature, 데이터 fingerprint, 선택 ID 순서를 비교합니다.
4. 각 문제 직전 실제 프롬프트/이미지 해시를 preflight와 비교합니다. 불일치하면 중단합니다.
5. 결과 보고 전에 실제 예측/입력/선택 ID 전체를 대조하고 채점/summary를 재검산합니다.
6. Pro 세 설정은 동일한 ID 집합이어야 하며 네 실행의 checkpoint와 source hash도 같아야 합니다.

Hub 데이터는 기존 revision을 고정합니다. 로컬은 해당 revision 이름의 snapshot 경로만 받으며 실제 출처는 사용자가 보존해야 합니다.
문항 수 검증만으로 로컬 파일이 공식 pinned snapshot임을 증명할 수는 없습니다.
모델/데이터 캐시가 있으면 재사용합니다. 환경 재설치나 드라이버 변경은 하지 않습니다.

`manifest.json`, `summary.json`, `selected_ids.json`, `inputs.jsonl`, `predictions.jsonl`,
`evaluation_profile.json`, `chat_template.txt`, `requirements.freeze.txt`, `environment.txt`,
`report_draft.md`를 각 실행별로 기록합니다. 전체 suite는 `summary.tsv`, `baseline_summary.md/json`을 생성합니다.
시간은 두 단계의 inference 합계와 모델 로딩/데이터 처리를 포함한 total을 구분합니다.
VRAM은 nvidia-smi 1초 샘플의 기기 전체 peak이며 다른 프로세스가 포함될 수 있습니다.

`--resume`는 검증된 완료 폴더를 건너뛰며, 미완료 결과를 성공으로 취급하지 않습니다.
이전 free 및 1024-token 결과는 변경하거나 재사용하여 새 4096 결과처럼 표시하지 않습니다.

## 이후 연구와의 관계

이 baseline 위에서 시각적 근거 활용, 다중 이미지 참조, 이미지/텍스트 배치에 대한 학습 개입을 비교합니다.
standard 조건도 이미지 번호·순서·텍스트 참조가 있어 연구 대상이며, vision만의 주제가 아닙니다.
레이아웃 증강을 학습에 적용하는 실험과 평가 입력 자체를 바꾸는 stress test를 구분합니다.
학습 후 메인 평가는 checkpoint만 바꾸고 이 평가 설정을 그대로 사용합니다.

MMMU-Pro test 전체의 기존 토큰/출력정책 실험을 이미 확인했으므로 이번 프로토콜은 그 실험의 영향을 받았습니다.
이 이력은 exploratory test ablation으로 공개하고, untouched test라고 부르지 않습니다.
이후 학습 데이터 선정/하이퍼파라미터 결정에 Pro 개별 정오답을 이용하지 않으며 개발 데이터에서 결정합니다.
모델 제출/학습 데이터 구성/최종 unseen 평가까지 이번 실행 하나로 완료되는 것은 아닙니다.

## 로컬 검증 범위

GPU 없는 단위 테스트로 MC/open 생성 경로, 공식 open 채점, 누락/중복/유형/설정 거부,
네 평가 6090행의 합성 결과 검증 및 보고서 생성을 점검합니다. 합성 결과는 실제 성능 수치가 아닙니다.
실제 4096 GPU 동작, 모든 실데이터 이미지의 입력 길이, 실행 시간/VRAM은 서버 실행에서 확인해야 합니다.
Context를 초과하는 입력은 조용히 토큰 예산을 줄이지 않고 실패시켜 설정 변경을 명시하도록 합니다.
