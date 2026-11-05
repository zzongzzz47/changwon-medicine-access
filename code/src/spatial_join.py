"""공개 격자 통계를 행정구역과 연결한다. 경계 절단 시 원래 격자 면적을 쓴다."""
from __future__ import annotations

import geopandas as gpd
import numpy as np


def allocate_grid_to_areas(grids, areas, value_columns=("population", "households")):
    """경계를 걸치는 격자의 공개값을 면적 비율로 나눈다.

    인구가 격자 안에 균일하게 분포한다는 근사다. 경계로 잘린 면적을
    분모로 사용하지 않는다. 거리 대표점은 절단 전 격자의 중심점이다.
    결측 통계는 결측으로 남기며, 실제 인구 0으로 해석하지 않는다.
    """
    if grids.crs is None or areas.crs is None:
        raise ValueError("격자와 행정경계의 좌표계가 필요합니다.")
    grids = grids.to_crs(5179).copy()
    areas = areas.to_crs(5179).copy()
    if not grids["grid_id"].is_unique or not areas["adm_cd"].is_unique:
        raise ValueError("격자 코드 또는 행정구역 코드가 중복됩니다.")
    if not grids.geometry.is_valid.all() or not areas.geometry.is_valid.all():
        raise ValueError("유효하지 않은 경계 도형이 있어 공간 연결을 중단합니다.")
    grids["original_area_m2"] = grids.area
    if (grids["original_area_m2"] <= 0).any():
        raise ValueError("격자 면적은 양수여야 합니다.")
    grids["x_m"] = grids.centroid.x
    grids["y_m"] = grids.centroid.y
    for col in value_columns:
        vals = grids[col].dropna()
        if not np.isfinite(vals).all() or (vals < 0).any():
            raise ValueError(f"{col} 공개 통계값의 범위를 확인하세요.")
    overlay = gpd.overlay(grids, areas, how="intersection", keep_geom_type=True)
    overlay = overlay.loc[overlay.area > 1e-6].copy()
    overlay["area_fraction"] = overlay.area / overlay["original_area_m2"]
    summed = overlay.groupby("grid_id")["area_fraction"].sum()
    if (summed > 1 + 1e-6).any():
        raise ValueError("행정경계가 중첩되어 같은 격자 값이 이중 배분됩니다.")
    for col in value_columns:
        overlay[col + "_reported"] = overlay[col]
        overlay[col + "_observed"] = overlay[col].notna()
        overlay[col] = overlay[col] * overlay["area_fraction"]
    overlay["demand_id"] = overlay["grid_id"].astype(str) + ":" + overlay["adm_cd"].astype(str)
    return overlay
