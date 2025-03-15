import pandas as pd
import numpy as np
import re

# 输入和输出文件路径
input_file_path = 'data/count_observation_upload.csv'  # 输入文件路径
output_file_path = 'data/video_data.csv'     # 输出文件路径

try:
    # 读取CSV文件
    df = pd.read_csv(input_file_path)
    
    # # 添加过滤条件，仅保留 index < 300 的记录
    # filtered_df = df[df['index'] <= 300]

    # 提取所需列，并填充缺失值为0
    extracted_columns = ['index', 'videoId', 'viewCount', 'viewCount_diff']
    extracted_df = df[extracted_columns].fillna(0)
    # 将 'viewCount_diff' 列中所有负数值替换为 0
    extracted_df['viewCount_diff'] = extracted_df['viewCount_diff'].apply(lambda x: max(0, x))

    # 保存为新的CSV文件
    extracted_df.to_csv(output_file_path, index=False)
    print(f"数据已成功提取并保存到: {output_file_path}")
    
    # # 创建 NumPy 数组，表示每种 index 对应的 viewCount_diff 值
    # extracted_df = extracted_df[['videoId', 'viewCount_diff']]
    # # 创建一个字典，存储每种视频的 viewCount_diff 列表
    # video_viewcount_diff_dict = {}
    # for video_id in video_ids:
    #     video_data = extracted_df[extracted_df['videoId'] == video_id]
    #     video_viewcount_diff_dict[video_id] = video_data['viewCount_diff'].tolist()
    
    # # 转换为 NumPy 数组（按行排列每个视频的 viewCount_diff）
    # viewcount_diff_array = np.array(list(video_viewcount_diff_dict.values()), dtype=object)
    
    # # 打印结果
    # print("每种视频的 viewCount_diff 数组：")
    # print(viewcount_diff_array.shape)
    # np.save('data/arr.npy', viewcount_diff_array)

except FileNotFoundError:
    print(f"错误: 文件 {input_file_path} 未找到，请检查路径和文件名是否正确。")
except Exception as e:
    print(f"发生错误: {str(e)}")