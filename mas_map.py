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
full_map = full_map.sort_values(by=['area', 'x', 'y'])
full_map.columns = full_map.columns.str.strip() #혹시나 모를 앞뒤 공백 없애기 (strip())
# 전체 병합된 full_map 출력
print("\n=== 모든 파일 병합 결과 (full_map) ===")
print(full_map)

# area가 1인 데이터만 추출
area1_data = full_map[full_map['area'] == 1]

full_map.to_csv('full_map.csv', index = False)

# area == 1인 데이터 출력
print("\n=== area == 1인 지역만 필터링한 결과 ===")
print(area1_data)










# 데이터 불러오기
df = full_map

# Construction Site가 표시된 구획 처리
df.loc[df['ConstructionSite'] == 1, 'struct'] = 'Construction Site'

# 구조물 이름이 비어있는 경우 '기타'로 채우기
df['struct'] = df['struct'].fillna('기타')

# 내 집 좌표에 'My House' 지정 (x=14, y=2 기준)
df.loc[(df['x'] == 14) & (df['y'] == 2), 'struct'] = 'My House'

# 구조물 종류 영문 표기로 통일
# name_map = {
#     'Apartment': 'Apartment',
#     'Building': 'Buildings',
#     'BandalgomCoffee': 'Bandalgom_Cafe',
#     'Construction Site': 'Construction Site',
#     'My House': 'My House'
# }
# df['struct'] = df['struct'].replace(name_map)




# 전체 구조물 수
total_count = len(df)

# 전체 건축물 구획수 (공사현장 제외)
non_construction = df[df['struct'] != 'Construction Site']


construction = df[df['category'] != 0 ]
total_buildings = len(construction)


# 구조물 종류별 통계
total_stats = df['struct'].value_counts()


df.to_csv('let_s_see.csv', index = False)

# 전역 통계 출력
print('[전역 구조물 현황]')
print(f'전체 구조물 갯수 = {total_count}개')
print(f'전체 건축물 구획수 = {total_buildings}개')
print(f'전체 Construction Site 구획수 = {total_stats.get("Construction Site", 0)}개')
print(f'Apartment 갯수 = {total_stats.get(" Apartment", 0)}개')
print(f'Buildings 갯수 = {total_stats.get(" Building", 0)}개')
print(f'My House 갯수 = {total_stats.get("My House", 0)}개')
print(f'Bandalgom_Cafe 갯수 = {total_stats.get(" BandalgomCoffee", 0)}개\n')

# Area별 요약 출력 함수
def print_area_summary(area_number):
    area_df = df[df['area'] == area_number]
    stats = area_df['struct'].value_counts()
    building_count = len(area_df[area_df['struct'] != 'Construction Site'])
    construction_count = stats.get('Construction Site', 0)

    print(f'[Area {area_number} 구조물 현황]')
    print(f'건축물 구획수 = {building_count}개')
    print(f'Construction Site 구획수 = {construction_count}개')
    if stats.get(' Apartment', 0):
        print(f'Apartment 갯수 = {stats[" Apartment"]}개')
    if stats.get(' Building', 0):
        print(f'Buildings 갯수 = {stats[" Building"]}개')
    if stats.get('My House', 0):
        print(f'My House 갯수 = {stats["My House"]}개')
    if stats.get(' BandalgomCoffee', 0):
        print(f'Bandalgom_Cafe 갯수 = {stats[" BandalgomCoffee"]}개')
    print('')

# 모든 구역 반복 출력
for area in sorted(df['area'].unique()):
    print_area_summary(area)