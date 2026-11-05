"""필수 실제 입력이 준비된 뒤 실행. 예시 인구로 결과를 만들지 않는다."""
import hashlib
import json
from pathlib import Path
from src.coverage_model import run_scenario

ROOT = Path(__file__).resolve().parent


def main():
    config_path = ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mode = config.get("scenario_mode", "convenience_only")
    if mode not in {"convenience_only", "supplement_existing"}:
        raise ValueError("scenario_mode는 convenience_only 또는 supplement_existing이어야 합니다.")
    keys = ["demand", "candidates"] + (["existing"] if mode == "supplement_existing" else [])
    paths = {k: ROOT / config[k] for k in keys}
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("아직 확보하지 못한 필수 입력: " + ", ".join(missing) +
                         "\nREADME의 자료 확보 순서와 data/inputs/입력규격.md를 확인하세요.")
    if not config.get("candidate_feasibility_reviewed"):
        raise SystemExit("후보지 영업상태·설치 적합성 검토 후 config 확인값을 설정하세요.")
    if not config.get("population_boundary_and_year_reviewed"):
        raise SystemExit("인구와 경계의 기준시점·단위를 검토한 뒤 config 확인값을 설정하세요.")
    data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    if mode == "convenience_only":
        data["existing"] = []
    result = {
        "config": config,
        "input_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
        "scenario_mode": mode,
        "limitations": ["편의점으로 독립적인 수거망을 구성하는 가정" if mode == "convenience_only"
                         else "기존 등록 수거함이 모두 이용 가능한 보완 시나리오",
                        "운영시간·동의·실제 수거량 미검증", "직선거리 기준이며 보행 네트워크 미반영"],
        "scenarios": [run_scenario(**data, radius_m=r, max_new=p)
                      for r in config["radii_m"] for p in config["max_new_options"]],
    }
    out = ROOT / "results" / "scenarios.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {out}")


if __name__ == "__main__":
    main()
