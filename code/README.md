# 사용한 코드

최종 분석의 중심은 `run_supplement_development.py`와 `src/supplement_model.py`입니다.

| 단계 | 스크립트 |
|---|---|
| 원본 공간 연결 | `build_spatial_inputs.py` |
| 초기 공간 분석 | `run_spatial_optimization.py` |
| 기존 수거망 진단 | `analyze_existing_network.py` |
| 후보 검토 | `run_candidate_audit.py` |
| 기존망 보완 최적화 | `run_supplement_development.py` |
| 거리 민감도 | `run_radius_review.py` |
| 결과 시각화 | `render_supplement_development.py`, `render_accessibility_extension.py` |

```sh
cd code
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

재계산에는 `data/raw`, `data/processed` 등의 원본·연결 자료가 필요합니다. 저장소에는 원자료를 포함하지 않았습니다. `../docs/DATA.md`와 각 스크립트의 입력 경로를 먼저 확인하세요. `run_operational_demo.py` 등 운영 확장 예제는 가상 조건이며 최종 결과와 구분합니다.
