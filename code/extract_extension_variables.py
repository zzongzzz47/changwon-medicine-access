"""Audit public input granularity; do not manufacture fine-grained demand."""
from pathlib import Path
from collections import Counter
import csv, json, hashlib
import openpyxl

P = Path(__file__).resolve().parent
R = P / 'results/accessibility_extension'

def write_csv(name, rows):
    with (R / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main():
    R.mkdir(parents=True, exist_ok=True)
    health_path = P / 'data/raw/nhis_gyeongnam_medical_use_2024.xlsx'
    w = openpyxl.load_workbook(health_path, read_only=True, data_only=True)
    def changwon(sheet):
        return {str(r[0]).replace('창원시 ', ''): (i, r)
                for i, r in enumerate(w[sheet].iter_rows(values_only=True), 1)
                if r[0] and str(r[0]).startswith('창원시 ')}
    main_rows = changwon('636p'); pop = changwon('470p')
    long_h = changwon('8-1(4)'); long_d = changwon('8-2(4)')
    assert len(main_rows) == 5
    health = []
    for district, (row_i, r) in main_rows.items():
        health.append({'district': district, 'year': 2024,
                       'medical_security_population': pop[district][1][1],
                       'hypertension_patients': r[1], 'hypertension_reimbursed_days': r[3],
                       'diabetes_patients': r[6], 'diabetes_reimbursed_days': r[8],
                       'hypertension_patients_360plus_days': long_h[district][1][6],
                       'diabetes_patients_360plus_days': long_d[district][1][6],
                       'disease_source_sheet_row': f'636p!{row_i}',
                       'population_source_sheet_row': f'470p!{pop[district][0]}',
                       'long_hypertension_source_sheet_row': f'8-1(4)!{long_h[district][0]}',
                       'long_diabetes_source_sheet_row': f'8-2(4)!{long_d[district][0]}'})
    write_csv('nhis_district_context_2024.csv', health)
    w.close()

    bus_path = P / 'data/raw/changwon_boardings_20250519_0525.xlsx'
    bus = openpyxl.load_workbook(bus_path, read_only=True, data_only=True)
    rows = [r for r in bus.worksheets[0].iter_rows(min_row=2, values_only=True) if r[0] is not None]
    locations_path = P / 'data/raw/changwon_bus_20251231.csv'
    with locations_path.open(encoding='utf-8-sig') as f:
        locations = list(csv.DictReader(f))
    # Exact names after surrounding whitespace only; direction ambiguity is retained.
    counts_a = Counter(str(r[1]).strip() for r in rows)
    counts_b = Counter(r['정류소명'].strip() for r in locations)
    unique_locations = {r['정류소명'].strip(): r for r in locations if counts_b[r['정류소명'].strip()] == 1}
    joined = []
    for r in rows:
        name = str(r[1]).strip()
        ok = counts_a[name] == counts_b[name] == 1
        loc = unique_locations[name] if ok else None
        status = 'unique_exact_name_provisional' if ok else ('no_exact_name' if counts_b[name] == 0 else 'ambiguous_name')
        joined.append({'boarding_stop_code': str(r[0]), 'name': name,
                       'nontransfer_events': int(r[2] or 0), 'transfer_events': int(r[3] or 0),
                       'boarding_events': int(r[2] or 0) + int(r[3] or 0), 'match_status': status,
                       'location_stop_id': loc['정류소아이디'] if ok else '',
                       'lon': float(loc['경도']) if ok else '', 'lat': float(loc['위도']) if ok else ''})
    assert len({r['boarding_stop_code'] for r in joined}) == len(joined)
    write_csv('boarding_stop_linkage_audit.csv', joined)
    hourly = {i: {'hour': i, 'nontransfer_events': 0, 'transfer_events': 0} for i in range(24)}
    for s in bus.worksheets[1:]:
        assert s.cell(1, 1).value == '시간대(1시간)', s.title
        for r in s.iter_rows(min_row=2, values_only=True):
            if not r[0]: continue
            hr = int(str(r[0]).split(':')[0])
            hourly[hr]['nontransfer_events'] += int(r[1] or 0)
            hourly[hr]['transfer_events'] += int(r[2] or 0)
    for r in hourly.values(): r['boarding_events'] = r['nontransfer_events'] + r['transfer_events']
    write_csv('boarding_route_hour_totals.csv', list(hourly.values()))
    matched = [r for r in joined if r['match_status'] == 'unique_exact_name_provisional']
    total = sum(r['boarding_events'] for r in joined)
    route_total = sum(r['boarding_events'] for r in hourly.values())
    after18 = sum(r['boarding_events'] for h, r in hourly.items() if h >= 18)
    out = {
        'health': {'year': 2024, 'spatial_units': 5, 'rows': health,
                   'use': 'district context only; no grid disaggregation or optimization weights',
                   'limitations': ['Patient counts may overlap across diseases; do not sum as unique people.',
                                   'Reimbursed days are not unused medicine quantities.',
                                   'District counts do not identify within-district patient locations.',
                                   'Age-adjusted rates and residence attribution are not evaluated in this extraction.']},
        'bus': {'period': '2025-05-19/2025-05-25', 'location_date': '2025-12-31',
                'stop_rows': len(joined), 'location_rows': len(locations), 'route_sheets': len(bus.worksheets)-1,
                'direct_id_matches': len(set(str(r[0]) for r in rows) & set(r['정류소아이디'] for r in locations)),
                'name_match_status_counts': dict(Counter(r['match_status'] for r in joined)),
                'unique_name_match_pct': 100*len(matched)/len(joined),
                'stop_boarding_events': total, 'matched_boarding_events': sum(r['boarding_events'] for r in matched),
                'matched_boarding_event_pct': 100*sum(r['boarding_events'] for r in matched)/total,
                'route_hour_boarding_events': route_total, 'stop_route_totals_equal': total == route_total,
                'route_events_18to24': after18, 'route_events_18to24_pct': 100*after18/route_total,
                'limitations': ['Boardings count repeated trips and transfers, not distinct residents.',
                               'Stop totals have no hour dimension; hour tables are route-level.',
                               'Only seven days are observed, so seasonality is not measured.',
                               'Unique exact-name joins are provisional because dates and identifier systems differ.',
                               'Do not optimize using the incomplete matched sample.']},
        'sha256': {str(p.relative_to(P)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [health_path, bus_path, locations_path]}}
    (R/'variable_data_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in out['bus'].items() if k != 'limitations'}, ensure_ascii=False))

if __name__ == '__main__': main()
