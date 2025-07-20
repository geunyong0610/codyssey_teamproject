import pandas as pd

# CSV 파일을 읽어옵니다.
area_map = pd.read_csv('area_map.csv')
area_struct = pd.read_csv('area_struct.csv')
area_category = pd.read_csv('area_category.csv')

# area_map 출력
print("=== area_map.csv ===")
print(area_map)

# area_struct 출력
print("\n=== area_struct.csv ===")
print(area_struct)

# area_category 출력
print("\n=== area_category.csv ===")
print(area_category)

# area_struct와 area_category를 category를 기준으로 병합
struct_with_name = pd.merge(
    area_struct, area_category,
    how="left",               # area_struct를 기준으로 병합 (왼쪽 기준)
    left_on="category",
    right_on="category"
)

# 병합된 struct_with_name 출력
print("\n=== area_struct + area_category (category 기준 병합) ===")
print(struct_with_name)

# struct_with_name과 area_map을 x, y 좌표 기준으로 병합
full_map = pd.merge(
    struct_with_name, area_map,
    how="left",
    on=['x', 'y']
)

# area 기준으로 정렬
full_map = full_map.sort_values(by='area')

# 전체 병합된 full_map 출력
print("\n=== 모든 파일 병합 결과 (full_map) ===")
print(full_map)

# area가 1인 데이터만 추출
area1_data = full_map[full_map['area'] == 1]

# area == 1인 데이터 출력
print("\n=== area == 1인 지역만 필터링한 결과 ===")
print(area1_data)
