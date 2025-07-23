import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from itertools import permutations
 #보너스 과제 부분


# 지정된 경로에서 csv 파일 불러와 pandas DataFrame으로 반환
def load_map_data(csv_path='sorted_raw_map.csv'):
    return pd.read_csv(csv_path)


# 구조물 이름으로 위치 찾는 함수(용도:시작점, 도착점 찾아서 좌표 반환)
def find_point(df, struct_name):
    df['struct'] = df['struct'].str.strip() #공백문자 제거
    point = df[df['struct'] == struct_name] # 인수로 데이터 프레임(df), 구조 이름을 받아서 해당 df에서 struct_name인 구조물의 위치만 필터링 
    if point.empty:
        raise ValueError(f'{struct_name} 위치를 찾을 수 없습니다.')
    
    return int(point.iloc[0]['x']), int(point.iloc[0]['y']) 


#좌표(nx, ny)가 유효한 이동 위치인지 검사
def is_valid(nx, ny, visited, grid, max_x, max_y):
    return (
        0 <= nx < max_x and
        0 <= ny < max_y and     #격자 x,y 범위 내
        not visited[nx][ny] and #방문한 적 없고
        grid[nx][ny] == 0       #공사구역 아닐 때
    )

# ConstructionSite를 저장하는 격자 생성하는 함수
def create_grid(df):
    max_x = df['x'].max() +1 
    max_y = df['y'].max() +1
    grid = [[0] * max_y for _ in range(max_x)]

    for _, row in df.iterrows():
        x, y = int(row['x']), int(row['y'])
        if int(row['ConstructionSite']) == 1:
            grid[x][y] = 1  # constructionsite: 못 지나감
    return grid



#최단거리 알고리즘
def bfs(start, end, grid):
    from_x, from_y = start  #시작 좌표
    to_x, to_y = end        #도착 좌표
    max_x, max_y = len(grid), len(grid[0]) #2차원 배열의 행/열의 개수
    
    visited = [[False] * max_y for _ in range(max_x)] 
    #방문 여부를 저장하는 2차원 배열
    
    prev = [[None] * max_y for _ in range(max_x)] 
    #이전 좌표 저장(경로 추적용)할 배열
    #prev[x][y]는 좌표 (x,y)오기 직전 어디서 왔는지 저장
    
    #큐 초기화
    queue = deque()
    queue.append((from_x, from_y))
    visited[from_x][from_y] = True 

    while queue: 
        x, y = queue.popleft() #큐에서 현재 위치 꺼냄(탐색할 지점)
        if (x, y) == (to_x, to_y): #도착지일 경우 종료
            break

        # 상하좌우 탐색; 유효한 위치면 방문처리, 현재 좌표 저장, 큐에 추가
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, visited, grid, max_x, max_y):
                visited[nx][ny] = True
                prev[nx][ny] = (x, y)
                queue.append((nx, ny))

    # 역추적 -> 최단 경로 저장 위함
    path = []
    x, y = to_x, to_y
    while (x, y) != (from_x, from_y):
        if prev[x][y] is None:
            return []  # 유효한 경로 없음
        path.append((x, y))
        x, y = prev[x][y]

    path.append((from_x, from_y)) 
    path.reverse()
    return path 
    #시작점 포함 후 순서 뒤집어 최단경로 반환


# 경로를 데이터 프레임으로 저장하고 home_to_cafe.csv파일로 출력
def save_path_csv(path, filename='home_to_cafe.csv'):
    df = pd.DataFrame(path, columns=['x', 'y'])
    df.to_csv(filename, index=False)

#2단계 지도 그리기 + 빨간선 최단경로 그리기 
def draw_map(df, path, filename='map_final.png'):
    spread_map = df.pivot(index='y', columns='x', values='category')  # x, y 좌표 기반으로 category를 scatter plot로 변환

    legend_names = {1: "Apartment", 2: "Buildings", 3: "My House", 4: "Bandalgom_Cafe", 5: "Construction Site"}  # 카테고리별 이름
    color_map = {1: "saddlebrown", 2: "saddlebrown", 3: "green", 4: "green", 5: "gray"}  # 카테고리별 색상
    marker_map = {1: "o", 2: "o", 3: "^", 4: "s", 5: "s"}  # 카테고리별 모양

    plt.figure(figsize = (6, 6))  # 그래프 크기

    for cat in sorted(df["category"].unique()):
        if cat == 0:
            continue

        subset = df[df["category"] == cat]
        plt.scatter(
            subset["x"], subset["y"],
            c = color_map[cat],
            marker = marker_map[cat],
            label = legend_names[cat],
            edgecolors = 'black',
            s = 400
        )
    
    # 최단 경로 그리기 (빨간 선)
    #유효한 경로가 나오면 빨간 선으로 시각화
    if path: 
        path_x = [x for x, y in path]
        path_y = [y for x, y in path]
        plt.plot(path_x, path_y, c='red', linewidth=2, label='Path')

    # 축 및 기타 시각화 설정
    plt.title("Cafe Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(markerscale=0.6)

    plt.xticks(spread_map.columns)
    plt.yticks(spread_map.index)
    plt.xlim(0.5, 15.5)
    plt.ylim(0.5, 15.5)
    plt.grid(True, linestyle=":", color="black", alpha=1.0)
    plt.gca().invert_yaxis()

    df.to_csv('sorted_raw_map.csv', index=False)  # CSV 저장
    plt.savefig(filename, dpi=300)  # 이미지 저장
    print("=== Map saved as '{}' ===".format(filename))
    plt.show()  # 화면에 표시
    plt.close()

# ======================== 보너스 과제 시작 ============================

### **최적 경로 탐색 함수: `find_optimal_structure_path`**

#이 함수는 **외판원 문제(TSP)** 의 개념을 적용하여, 모든 지정된 구조물(시작점 제외)에 대한 **순열**을 생성하고, 각 순열에 대해 BFS를 사용하여 전체 경로 길이를 계산합니다. 가장 짧은 총 경로 길이를 가진 순열이 최적의 경로로 선택됩니다.

def find_optimal_structure_path(df, start_structure='MyHome', target_structures=['Building', 'Apartment', 'BandalgomCoffee']):
    """
    지정된 시작 구조물에서 시작하여 모든 목표 구조물을 정확히 한 번씩 방문하는 최적의 경로를 계산합니다.

    인자:
        df (pd.DataFrame): 지도 데이터 DataFrame.
        start_structure (str): 시작 구조물의 이름.
        target_structures (list): 방문할 구조물 이름 리스트.

    반환값:
        tuple: 다음을 포함하는 튜플:
            - list: 최적의 전체 경로 (좌표 리스트).
            - int: 최적 경로의 총 길이.
            - list: 최적의 구조물 방문 순서.
    """

    # 1. 모든 관련 구조물의 좌표를 가져오고 격자를 생성합니다.
    grid = create_grid(df)

    # 순열을 생성할 때 시작 구조물은 제외합니다. 고정된 시작점이기 때문입니다.
    # 나중에 경로 계산을 위해 각 순열의 시작 부분에 다시 삽입됩니다.
    structures_to_visit_names = [s for s in target_structures if s != start_structure]

    structure_coords = {}
    for struct_name in [start_structure] + structures_to_visit_names:
        try:
            structure_coords[struct_name] = find_point(df, struct_name)
        except ValueError as e:
            print(f"오류: {e}. CSV 파일의 구조물 이름을 확인해주세요.")
            return [], 0, []

    min_total_path_length = float('inf') # 최소 경로 길이를 무한대로 초기화
    optimal_full_path = []
    optimal_structure_sequence = []

    # 2. 목표 구조물들의 모든 가능한 순열을 생성합니다.
    for perm in permutations(structures_to_visit_names):
        current_sequence = [start_structure] + list(perm) # 시작 구조물을 포함한 현재 순서
        current_total_path_length = 0
        current_full_path = []

        # 현재 순열에 대한 경로를 계산합니다.
        for i in range(len(current_sequence) - 1):
            start_point = structure_coords[current_sequence[i]]
            end_point = structure_coords[current_sequence[i+1]]

            path_segment = bfs(start_point, end_point, grid)

            if not path_segment: # 경로를 찾을 수 없으면 이 순열은 유효하지 않음
                current_total_path_length = float('inf')
                break

            # 경로 세그먼트의 길이를 더합니다.
            # 첫 번째 세그먼트가 아니라면 연결점을 중복으로 세는 것을 피하기 위해 1을 뺍니다.
            current_total_path_length += (len(path_segment) - 1) if i > 0 else len(path_segment)

            # 전체 경로에 세그먼트를 추가합니다.
            # 첫 번째 세그먼트가 아니면 시작점을 제외하고 추가하여 중복을 피합니다.
            if i == 0:
                current_full_path.extend(path_segment)
            else:
                current_full_path.extend(path_segment[1:])

        # 현재 경로가 이전의 최소 경로보다 짧으면 최적 경로를 업데이트합니다.
        if current_total_path_length < min_total_path_length:
            min_total_path_length = current_total_path_length
            optimal_full_path = current_full_path
            optimal_structure_sequence = current_sequence

    return optimal_full_path, min_total_path_length, optimal_structure_sequence

# === 실행 부분 ===

if __name__ == '__main__':
    # 지도 데이터 불러오기
    df = load_map_data()

    # 시작 지점과 방문할 구조물 목록 정의
    start_point_name = 'MyHome'
    structures_to_visit = ['Building', 'Apartment', 'BandalgomCoffee']

    # 최적 경로 찾기
    optimal_path, total_length, optimal_sequence = find_optimal_structure_path(
        df,
        start_structure=start_point_name,
        target_structures=structures_to_visit
    )

    if optimal_path:
        print(f"최적 경로를 찾았습니다! 총 길이: {total_length} 단위.")
        print(f"최적 구조물 방문 순서: {optimal_sequence}")

        # 경로를 CSV 파일로 저장 (선택 사항)
        save_path_csv(optimal_path, 'optimal_full_tour_path.csv')

        # 최적 경로를 포함하여 지도 그리기
        draw_map(df, optimal_path, 'optimal_tour_map.png')
    else:
        print("모든 구조물을 방문할 유효한 경로를 찾을 수 없습니다.")

        
# def main():
#     df = load_map_data()
#     start = find_point(df, 'MyHome')
#     end = find_point(df, 'BandalgomCoffee')

#     grid = create_grid(df)
#     path = bfs(start, end, grid)

#     if not path:
#         print('경로를 찾을 수 없습니다.')
#         return

#     save_path_csv(path)
#     #draw_map(df, path)

#     #보너스 과제 
#     df = load_map_data()
#     path = find_greedy_struct_path(df)
#     save_path_csv(path, 'bonus_home_to_cafe.csv')
#     draw_map(df, path, 'bonus_map_final.png')


# if __name__ == '__main__':
#     main()
