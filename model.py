import os
import pandas as pd
from pathlib import Path


def convert_peaje(peaje_value):
    """将Peaje转换为6.2TD格式"""
    return "6.2TD"


def convert_periodo(periodo_value):
    """将Periodo转换为6P模式"""
    periodo_map = {
        '1': 'P1', '2': 'P2', '3': 'P6'
    }
    return periodo_map.get(str(periodo_value), f"P{periodo_value}")


def extract_pvpc_data(file_path):
    """
    从EF_pvpcdata_fechaini文件中提取指定字段数据
    """

    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 调试：显示文件前几行
        print(f"文件头信息:")
        for i, line in enumerate(lines[:3]):
            print(f"  第{i + 1}行: {line.strip()}")

        # 寻找数据行开始位置
        data_lines = []
        header_found = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 跳过文件头标识行
            if line.startswith('pvpcdata;'):
                header_found = True
                continue

            # 跳过可能的时间戳行（包含多个数字和分号）
            if header_found and ';' in line and len(line.split(';')) > 3:
                # 检查是否是数据行（以日期格式开头）
                if line.startswith(('01/', '02/', '03/', '04/', '05/', '06/',
                                    '07/', '08/', '09/', '10/', '11/', '12/', '13/',
                                    '14/', '15/', '16/', '17/', '18/', '19/', '20/',
                                    '21/', '22/', '23/', '24/', '25/', '26/', '27/',
                                    '28/', '29/', '30/', '31/')):
                    data_lines.append(line)

        print(f"找到 {len(data_lines)} 个数据行")

        if not data_lines:
            print("警告: 未找到有效的数据行")
            return None

        extracted_data = []
        valid_records = 0

        for line_num, line in enumerate(data_lines, 1):
            line = line.strip()
            # 移除行尾的 * 号（如果有）
            if line.endswith('*'):
                line = line[:-1]
            if line.endswith(';'):
                line = line[:-1]

            fields = line.split(';')

            print(f"第{line_num}行有 {len(fields)} 个字段")

            # 检查字段数量是否足够
            if len(fields) < 50:  # 我们需要至少50个字段
                print(f"警告: 第{line_num}行字段不足 ({len(fields)}/50)，跳过")
                continue

            try:
                record = {
                    'Time': fields[0],  # 字段1
                    'Hour': fields[1],  # 字段2
                    'Peaje_Org': fields[2],  # 字段3
                    'Peaje_Trans': convert_peaje(fields[2]),
                    'Periodo_Org': fields[3],  # 字段4
                    'Periodo_Trans': convert_periodo(fields[3]),
                    'PMD': fields[36],  # 字段37
                    'Ai1': fields[48],  # 字段49
                    'Ai2': fields[49],  # 字段50
                    'Bi': fields[26],  # 字段27
                    'Perd': fields[5],  # 字段6
                    'Green': fields[30],  # 字段31
                    'TEAr1': fields[8],  # 字段9
                    'TEAr2': fields[9]  # 字段10
                }

                # 验证关键字段不为空
                if record['Time'] and record['Hour']:
                    extracted_data.append(record)
                    valid_records += 1
                else:
                    print(f"警告: 第{line_num}行关键字段为空，跳过")

            except IndexError as e:
                print(f"错误: 第{line_num}行字段索引错误: {e}")
                continue
            except Exception as e:
                print(f"错误处理第{line_num}行: {e}")
                continue

        print(f"成功提取 {valid_records} 条有效记录")

        if extracted_data:
            df = pd.DataFrame(extracted_data)

            # 重新排列列的顺序
            column_order = [
                'Time', 'Hour', 'Peaje_Trans', 'Periodo_Trans',
                'PMD', 'Ai1', 'Ai2', 'Bi', 'Perd', 'Green',
                'TEAr1', 'TEAr2', 'Peaje_Org', 'Periodo_Org'
            ]

            available_columns = [col for col in column_order if col in df.columns]
            df = df[available_columns]

            return df
        else:
            print("没有提取到任何有效记录")
            return None

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def process_all_pvpc_files(data_folder="Data"):
    """处理Data文件夹中的所有PVPC数据文件"""

    data_path = Path(data_folder)

    if not data_path.exists():
        print(f"错误: 文件夹 {data_folder} 不存在")
        return

    # 查找所有PVPC数据文件
    pvpc_patterns = [
        "EF_pvpcdata_*",
        "A1_pvpcdata_*",
        "C2_pvpcdata_*",
        "pvpcdata_*"
    ]

    pvpc_files = []
    for pattern in pvpc_patterns:
        pvpc_files.extend(list(data_path.glob(pattern)))

    # 去重
    pvpc_files = list(set(pvpc_files))

    if not pvpc_files:
        print(f"在文件夹 {data_folder} 中未找到PVPC数据文件")
        return

    print(f"找到 {len(pvpc_files)} 个PVPC数据文件")

    all_data = []
    processed_files = 0

    for file_path in sorted(pvpc_files):
        print(f"\n{'=' * 50}")
        print(f"处理文件 ({processed_files + 1}/{len(pvpc_files)}): {file_path.name}")
        print(f"{'=' * 50}")

        # 提取数据
        df = extract_pvpc_data(file_path)

        if df is not None and not df.empty:
            # 添加文件名作为标识
            df['源文件'] = file_path.name
            all_data.append(df)
            processed_files += 1

            # 显示前几行数据
            print(f"成功提取 {len(df)} 条记录")
            print("前3行数据预览:")
            print(df.head(3).to_string(index=False))

            # 保存单个文件的提取结果
            output_file = file_path.stem + "_extracted.csv"
            df.to_csv(data_path / output_file, index=False, encoding='utf-8-sig')
            print(f"数据已保存到: {output_file}")
        else:
            print(f"无法从文件 {file_path.name} 提取数据")

    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # 显示汇总信息
        print(f"\n数据汇总:")
        print(f"成功处理文件: {processed_files}/{len(pvpc_files)}")
        print(f"总记录数: {len(combined_df)}")
        if not combined_df.empty:
            print(f"日期范围: {combined_df['Time'].min()} 到 {combined_df['Time'].max()}")

        return combined_df
    else:
        print("没有成功提取任何数据")
        return None


# 主程序
if __name__ == "__main__":
    print("开始提取PVPC数据文件...")

    # 处理所有PVPC文件
    result_df = process_all_pvpc_files("Data")

    if result_df is not None:
        print(f"\n最终结果: 成功处理 {len(result_df)} 条记录")
    else:
        print("\n最终结果: 未能提取任何数据")

    print("\n程序执行完成!")