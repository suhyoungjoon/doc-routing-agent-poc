# 프로젝트: 사내 수신 문서 분류·라우팅 에이전트 (PoC)

## 목적
사내로 들어오는 문서(`./sample_docs`)를 Claude API로 읽고, 설정된 카테고리로 분류하고,
필수 필드를 추출한 뒤, 승인 게이트를 거쳐 지정 경로로 라우팅한다.
문서 본문에 숨은 지시(프롬프트 인젝션)에 넘어가지 않는 것이 핵심 검증 포인트.

## 파이프라인
```
connector (read-only, sample_docs 화이트리스트)
  → defense (injection_defense.yaml 패턴 매칭 + on_detect)
  → classifier (Claude API, categories.yaml 기준 분류 + required_fields 추출)
  → gate (approval_thresholds.yaml 기준 승인 필요 여부)
  → router (categories.yaml route_to, PoC는 로그 시뮬레이션)
모든 단계 입력·판단·출력 → logs/traces/<run_id>.jsonl
```

## 절대 규칙
- **카테고리명·임계값·방어 패턴·경로는 코드에 직접 쓰지 않는다.**
  반드시 `src/config_loader.py`를 통해서만 참조한다. (설정 원본: `config/*.yaml`)
- connector는 read-only. `sample_docs` 밖 경로(`..`, 절대경로, 심볼릭 링크 탈출)는 예외.
- 새 모듈은 위반/실패 케이스 테스트를 먼저 작성한 뒤 구현한다.

## 구조
```
config/     categories.yaml, injection_defense.yaml, approval_thresholds.yaml, eval_criteria.yaml
src/        config_loader.py, connector.py, classifier.py, tracing.py, defense.py, gate.py, router.py
eval/       test_cases.jsonl, runner.py
sample_docs/  입력 문서
logs/traces/  실행 트레이스 (gitignore)
tests/      pytest
```

## 실행
```
uv sync
uv run python src/config_loader.py   # 설정 로드 검증
uv run pytest
```
환경변수: `ANTHROPIC_API_KEY` (classifier), `DOC_AGENT_CONFIG_DIR` (선택, 기본 `./config`)
