import pandas as pd #pandas 라이브러리 불러오기
import matplotlib.pyplot as plt #matplotlib.pyplot 라이브러리 불러오기

# 우선 파일을 읽어준 뒤에 건설현장을 카테고리 "5"로 변경하는 부분!

raw_map = pd.read_csv("full_map.csv ") #full_map_sorted.csv 파일 읽어오기

raw_map.columns = raw_map.columns.str.strip() #혹시 모를 index 부분의 공백 (예시: " struct") 제거하기
cond = raw_map["ConstructionSite"] == 1 # "ConstructionSite" 컬럼이 1인 경우를 조건으로 설정. cond는 index값과 boolean 값으로 이루어진 시리즈로, 이후 loc[] 메소드를 사용하여 해당 조건을 만족하는 행을 선택할 수 있습니다.

#형식: .loc[조건, "시행할 column 이름"]  = 값 (건설현장= 카테고리 5로 변경)
raw_map.loc[cond, "category"] = 5 #category의 값을 ConstructionSite = 1인 경우 5로 일괄 변경해줍니다!

#같은 방식으로 "struct"의 값도 업데이트해줍니다.
raw_map.loc[cond, "struct"] = "ConSite" # "ConstructionSite"가 1인 경우 struct = ConSite로 변경!

# 두 값을 덮어씌우면서 "건설현장과 기타 구조물이 겹치는 경우 건설 현장으로 판단한다" 는 수행 내용 조건을 만족합니다.

# 분석된 데이터를 기반으로 지역 지도를 시각화하기

spread_map = raw_map.pivot(index='y', columns='x', values='category') # x, y 좌표 기반으로 category를 scatter plot로 변환 (각 x, y 위치에는 카테고리 값이 들어감)


# 우선 Category별로 이름, 색상 및 shape를 다르게 하여 시각화합니다.
legend_names = {1: "Apartment", 2: "Buildings", 3:"My House", 4: "Bandalgom_Cafe", 5: "Construction Site"} #카테고리별 이름을 지정해줍니다.
color_map = {1: "saddlebrown", 2: "saddlebrown", 3: "green", 4: "green", 5: "gray"} #그 뒤에 카테고리별 색상을 지정해주고, 
marker_map = {1: "o", 2: "o", 3: "^", 4: "s", 5: "s"} #카테고리별 모양을 지정해줍니다.


# 이제 plot을 해 보겠습니다. 
plt.figure(figsize = (6, 6)) # 그래프의 크기 설정

for cat in sorted(raw_map["category"].unique()): #현재 raw_map에 존재하는 카테고리 번호를 오름차순으로 리스트로 가져온 다음 일일히 이하 구문을 실행합니다.
        if cat == 0: # 카테고리 0 = 정의되지 않은 빈 공간이니
            continue # 건너뜁니다. (심볼, 색깔 없음)

        subset = raw_map[raw_map["category"] == cat] # 카테고리별로 서브셋을 만듭니다. (예: cat = 1인 경우 카테고리가 1인 건물을 모아놓은 subset의 x, y 좌표로 이루어진 데이터프레임)
        plt.scatter( #Scatter plot을 사용하여 시각화합니다. 어떻게?
            subset["x"], subset["y"], #x, y 좌표 바탕으로
            c = color_map[cat], # 카테고리별 색상
            marker = marker_map[cat], # 카테고리별 모양
            label = legend_names[cat], # 카테고리별 이름
            edgecolors = 'black', #테두리는 검은색!
            s = 400 #마커 크기는 400 픽셀로 설정
        )

# 그래프의 제목, 축의 이름, 축 눈금 및 보여줄 축의 범위, 범례 및 grid 등을 설정하는 부분입니다!

plt.title("Cafe Map") # 그래프 제목을 "Cafe Map"으로 설정
plt.xlabel("X") #X 축 이름
plt.ylabel("Y") #Y 축 이름
plt.legend(markerscale=0.6) #legend(범례) 추가. 마커의 스케일을 범례에 맞게 조정합니다.

plt.xticks(spread_map.columns)  # x축 눈금 설정
plt.yticks(spread_map.index)  # y축 눈금 설정
plt.xlim(0.5, 15.5) #x, y축이 보여주는 최소, 최대값을 설정함
plt.ylim(0.5, 15.5) # 각 축을 1, 15로 설정할 시 가장자리 symbol이 잘려서 좀 더 여유를 갖고 설정!
plt.grid(True, linestyle=":", color="black", alpha=1.0) #격자선 존재(그려준다는 의미), 점선(:)으로, 색깔은 검정색, alpha값(투명도)=불투명(1.0)
plt.gca().invert_yaxis()  # y축을 뒤집어줍니다. (0, 0)이 좌측 하단이 아닌 좌측 상단에 위치하도록 하기 위함입니다.

raw_map.to_csv('sorted_raw_map.csv', index=False)
# sorted_raw_map을 CSV 파일로 저장
plt.savefig('map.png', dpi=300)
# map.png 파일로 저장
print("=== Map saved as 'map.png' ===")
plt.show() #plot을 새 창에다가 띄워줍니다. 

#주의사항: plot 창을 닫아야 코드가 종료됩니다!
#본 프로그램은 실행 시 'full_map_sorted.csv' 파일이 같은 디렉토리에 있어야 합니다.
#본 프로그램은 실행시 .py파일이 있는 디렉토리에 map.png 파일을 생성합니다.
#본 프로그램은 실행시 .py파일이 있는 디렉토리에 sorted_raw_map.csv 파일을 생성합니다.

## 이후 3단계를 실행할 때는 .csv 파일의 struct 행 ConstructionSite 컬럼의 값을 
## ConSite로 수정하였으므로 이 점 참고하여 코딩하시기 바랍니다.