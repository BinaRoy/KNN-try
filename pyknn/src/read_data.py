import h5py

def explore_h5_file(filepath, output_filepath):
    with h5py.File(filepath, 'r') as file:
        with open(output_filepath, 'w') as f:
            print("Keys in the file:")
            f.write("Keys in the file:\n")
            for key in file.keys():
                print(key)
                f.write(f"{key}\n")
                explore_h5_group(file[key], f, indent=1)

def explore_h5_group(group, f, indent=1):
    """递归函数以展示嵌套结构并打印部分数据"""
    try:
        for key in group.keys():
            print('  ' * indent + key)
            f.write('  ' * indent + key + '\n')
            explore_h5_group(group[key], f, indent + 1)
    except AttributeError:
        # 如果不是组，就是数据集
        print('  ' * indent + f"[Dataset with shape: {group.shape}]")
        f.write('  ' * indent + f"[Dataset with shape: {group.shape}]\n")
        # 打印部分数据内容，例如前5个元素（如果数据集足够大）
        data_sample = group[:5] if group.size > 5 else group[:]
        print('  ' * indent + f"Data sample: {data_sample}")
        f.write('  ' * indent + f"Data sample: {data_sample}\n")

# 替换 'your_file.h5' 为你的实际文件路径
file_path = '/home/tkvr85/AMD/pyknn/data/topologically_stable_cores/topologically_stable_cores.h5'
output_path = '/home/tkvr85/AMD/pyknn/data/topologically_stable_cores/exploration_output.txt'
explore_h5_file(file_path, output_path)


