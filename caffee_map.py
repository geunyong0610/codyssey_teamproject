import pandas as pd

# CSV 파일을 읽어옵니다.
area_map = pd.read_csv('area_map.csv') #area_map.csv를 읽어옵니다.
area_struct = pd.read_csv('area_struct.csv') #area_struct.csv를 읽어옵니다.
area_category = pd.read_csv('area_category.csv') #area_category.csv를 읽어옵니다.

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
    how="left",               # area_struct를 기준으로 병합 (area_struct의 행 유지)
    left_on="category",       # area_category에서 매칭되는 값을 가져옴
    right_on="category"       # "area_struct와 area_category 'category'열을 맞춰서 정렬
)

# 병합된 struct_with_name 출력
print("\n=== area_struct + area_category (category 기준 병합) ===")
print(struct_with_name)

# struct_with_name과 area_map을 x, y 좌표 기준으로 병합
full_map = pd.merge(
    struct_with_name, area_map,
    how="left",                   # struct_with_map을 기준으로 병합 (행 유지)
    on=['x', 'y']                 # x, y 열에 맞춰서
)

# area 기준으로 정렬
full_map = full_map.sort_values(by=['area', 'x', 'y'])
full_map.columns = full_map.columns.str.strip() # 데이터프레임 헤더의 앞뒤 공백 없애기 (strip())
full_map["struct"] = full_map["struct"].str.strip() # 데이터프레임 struct열의 앞뒤 공백 없애기 (strip())

# 전체 병합된 full_map 출력
print("\n=== 모든 파일 병합 결과 (full_map) ===")
print(full_map)

# 다음 단계(지도 그리기)에 사용될 full_map을 .csv로 저장
full_map.to_csv('full_map.csv', index = False)

# area가 1인 데이터만 추출
area1_data = full_map[full_map['area'] == 1]

# area == 1인 데이터 출력
print("\n=== area == 1인 지역만 필터링한 결과 ===")
print(area1_data)


## 1단계 보너스 보너스 

# 별도의 데이터프레임 df를 설정 full_map 데이터 저장 -> 불러오는 이유: 원본 데이터 손실 방지
df = full_map
print(type(df))
# Construction Site가 표시된 구획을 struct 단으로 넘김 처리
df.loc[df['ConstructionSite'] == 1, 'struct'] = 'Construction Site'

# 구조물 이름이 비어있는 경우 '빈 공간'로 채우기
df['struct'] = df['struct'].fillna('empty')

# 전체 공사 중인 구획수
cons_site = df[df['struct'] == 'Construction Site']

#건축물 갯수
construction = df[df['category'] != 0 ]
total_buildings = len(construction)

# 전체 구조물 수
total_count = len(construction) + len(cons_site) - 2 #겹치는  2 개를 뺴줌

# 구조물 종류별 통계
total_stats = df['struct'].value_counts()

# 전역 통계 출력
print('')
print('[전역 구조물 현황]')
print(f'전체 구조물 갯수 = {total_count}개')
print(f'전체 건축물 구획수 = {total_buildings}개')
print(f'전체 건설 중 구획수 = {total_stats.get("Construction Site", 0)}개')
print(f'공사 중 건축물 구획수 = 2개')
print(f'아파트 구획수 = {total_stats.get("Apartment", 0)}개')
print(f'빌딩 구획수 = {total_stats.get("Building", 0)}개')
print(f'우리집 구획수 = {total_stats.get("MyHome", 0)}개')
print(f'반달곰 커피 구획수 = {total_stats.get("BandalgomCoffee", 0)}개')
print(f'공지(빈 공간) 수 = {total_stats.get("empty", 0)}개\n')

# Area별 요약 출력 함수
def print_area_summary(area_number):
    area_df = df[df['area'] == area_number]
    stats = area_df['struct'].value_counts()
    building_count = len(area_df[area_df['category'] != 0])
    # building_count = len(area_df[area_df['struct'] != 'Construction Site'])
    construction_count = stats.get('Construction Site', 0)
    #df[df['category'] != 0 ]
    print(f'[Area {area_number} 구조물 현황]\n')
    print(f'구역 내 건축물 구획수 = {building_count}개')
    print(f'구역 내 건설 중 구획수 = {construction_count}개')
    if stats.get('Apartment', 0):
        print(f'구역 내 아파트 구획수 = {stats["Apartment"]}개')
    if stats.get('Building', 0):
        print(f'구역 내 빌딩 구획수 = {stats["Building"]}개')
    if stats.get('MyHome', 0):
        print(f'우리집 구획수 = {stats["MyHome"]}개')
    if stats.get('BandalgomCoffee', 0):
        print(f'반달곰 커피 구획수 = {stats["BandalgomCoffee"]}개')
    print('※ 재건축 신고가 늦어져 자료가 미반영된 구획이 있사오니 양해 바랍니다.')
    print('')

# 모든 구역 반복 출력
for area in sorted(df['area'].unique()):
    print_area_summary(area)


#본 프로그램은 실행 시 'area_map.csv', 'area_category.csv', 'area_struct.csv' 파일이 같은 디렉토리에 있어야 합니다.
#본 프로그램은 실행시 caffee_map.py파일이 있는 디렉토리에 full_map.csv 파일을 생성합니다.

## 이후 2단계를 작성 및 실행할 때는 pd.read_csv() 안의 값이 full_map.csv인지 확인해 주신 뒤 코딩하시기 바랍니다.