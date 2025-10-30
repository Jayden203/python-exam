#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import os

# 设置 Matplotlib 样式
plt.style.use('ggplot')


# --- Part 3: 基本神经网络定义 (定义在函数外部以便清晰) ---
class SimpleNN(nn.Module):
    """
    一个简单的全连接神经网络，用于 Part 3。
    结构: 10输入 -> 5神经元 (ReLU) -> 3神经元 (ReLU) -> 1输出 (Sigmoid)
    """

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 输入层到隐藏层 1
        self.fc2 = nn.Linear(5, 3)  # 隐藏层 1 到隐藏层 2
        self.fc3 = nn.Linear(3, 1)  # 隐藏层 2 到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# --- Part 1: Python 基础和数据操作 ---
def part1_data_analysis(file_path):
    """
    执行数据加载、清洗和可视化任务 (Part 1)。
    返回清洗后的 DataFrame。
    """
    print("--- 任务 1: Python 基础和数据操作 ---")

    # Task 1a: 数据加载和预处理
    print("--- Task 1a: 数据加载和预处理 ---")
    try:
        # 1. 加载数据
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return pd.DataFrame()

    # 2. 数据清洗
    # 填充缺失值
    df['Product'].fillna('Unknown', inplace=True)
    df['Quantity'].fillna(0, inplace=True)
    df['Price'].fillna(0.0, inplace=True)

    # 转换 Date 列为 datetime 对象
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)  # 丢弃日期转换失败的行

    # 确保 Total 列正确
    df['Total'] = df['Quantity'] * df['Price']
    print(f"数据清洗完成。处理了 {len(df)} 条记录。")

    # Task 1b: 数据可视化
    print("\n--- Task 1b: 数据可视化 ---")

    # 1. 产品销量分布 (条形图)
    product_quantity = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    product_quantity.plot(kind='bar', color='skyblue')
    plt.title('Total Quantity Sold per Product )')
    plt.xlabel('Product ')
    plt.ylabel('Total Quantity Sold ')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('product_sales_distribution.png')
    print("图表 'product_sales_distribution.png' 已生成。")

    # 2. 随时间变化的销售额 (线图)
    df_2023 = df[df['Date'].dt.year == 2023].copy()
    if not df_2023.empty:
        df_2023.set_index('Date', inplace=True)
        monthly_sales_2023 = df_2023['Total'].resample('M').sum()
        monthly_sales_2023.index = monthly_sales_2023.index.strftime('%Y-%m')

        plt.figure(figsize=(12, 6))
        monthly_sales_2023.plot(kind='line', marker='o', linestyle='-', color='green')
        plt.title('Total Sales Over Time (2023) ')
        plt.xlabel('Month')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('sales_over_time_2023.png')
        print("图表 'sales_over_time_2023.png' 已生成。")
    else:
        print("警告：2023 年数据为空，未生成销售趋势图。")

    # 准备用于 Part 2 的 DataFrame
    df_for_db = df.copy()
    df_for_db['Date'] = df_for_db['Date'].dt.strftime('%Y-%m-%d')
    return df_for_db


# --- Part 2: Python 数据库管理 ---
def part2_database_management(df_cleaned):
    """
    执行 SQLite 数据库创建、数据插入和查询任务 (Part 2)。
    """
    if df_cleaned.empty:
        print("\n--- 任务 2: 数据库管理 ---")
        print("数据为空，跳过数据库操作。")
        return

    print("\n--- 任务 2: Python 数据库管理 ---")

    DB_NAME = 'SalesDB.sqlite'
    TABLE_NAME = 'Sales'

    # Task 2a: 数据库创建和数据插入
    print("--- Task 2a: 数据库创建和数据插入 ---")

    # 1. 创建 SQLite 数据库
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    print(f"已连接到数据库: {DB_NAME}")

    # 2. 创建 Sales 表
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        Date TEXT,
        Product TEXT,
        Quantity INTEGER,
        Price REAL,
        Total REAL,
        UNIQUE (Date, Product, Quantity, Price)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    print(f"表 '{TABLE_NAME}' 创建完成。")

    # 3. 插入数据
    # 使用 if_exists='replace' 确保每次运行都是全新的数据，满足“无重复项”的要求
    df_cleaned.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    print(f"已插入 {len(df_cleaned)} 行数据到 '{TABLE_NAME}' 表。")

    # Task 2b: 查询数据库
    print("\n--- Task 2b: 查询数据库 ---")

    # 1. 2023 年总销售额计算
    total_sales_2023_query = f"""
    SELECT SUM(Total)
    FROM {TABLE_NAME}
    WHERE Date LIKE '2023%';
    """
    cursor.execute(total_sales_2023_query)
    total_sales_2023 = cursor.fetchone()[0]
    print(f"1. 2023 年总销售额: ${total_sales_2023:.2f}")

    # 2. 2023 年产品销售概况
    product_sales_2023_query = f"""
    SELECT Product, SUM(Quantity) AS TotalQuantitySold
    FROM {TABLE_NAME}
    WHERE Date LIKE '2023%'
    GROUP BY Product
    ORDER BY TotalQuantitySold DESC;
    """
    cursor.execute(product_sales_2023_query)
    product_sales_summary = cursor.fetchall()

    print("\n2. 2023 年产品销售概况 (按数量降序):")
    for product, quantity in product_sales_summary:
        print(f"   - {product}: {int(quantity)} units")

    conn.close()


# --- Part 3: 基础神经网络实现 ---
def part3_neural_network():
    """
    执行 PyTorch 神经网络定义、训练和可视化任务 (Part 3)。
    """
    print("\n--- 任务 3: 基本神经网络实现 (PyTorch) ---")

    # Task 3a: 模型初始化
    model = SimpleNN()
    criterion = nn.MSELoss()  # 损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器

    print("--- Task 3a: 模型初始化 ---")
    print(f"模型: SimpleNN (10->5->3->1)")
    print(f"损失函数: {criterion.__class__.__name__}")
    print(f"优化器: {optimizer.__class__.__name__} (lr=0.01)")

    # Task 3b: 数据生成
    X = torch.randn(100, 10)  # 100 样本, 10 特征
    y = torch.randint(0, 2, (100, 1)).float()  # 100 目标值 (0 或 1)
    print("\n--- Task 3b: 数据生成 ---")
    print(f"合成数据生成完成: X shape {X.shape}, y shape {y.shape}")

    # Task 3c: 训练网络
    epochs = 20
    loss_history = []
    print("\n--- Task 3c: 训练网络 ---")
    print(f"开始训练 {epochs} 个周期...")

    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f'周期 [{epoch + 1}/{epochs}], 损失 (Loss): {loss.item():.4f}')

    print("训练完成。")

    # Task 3d: 损失可视化
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-', color='red')
    plt.title('Training Loss Over Epochs ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss_nn.png')
    print("\n--- Task 3d: 损失可视化 ---")
    print("图表 'training_loss_nn.png' 已生成。")


# --- 主执行函数 ---
def main():
    """
    执行所有三个任务的主函数。
    """
    SALES_FILE = 'sales_data.csv'

    # 1. 执行 Part 1: 数据分析
    cleaned_df = part1_data_analysis(SALES_FILE)

    # 2. 执行 Part 2: 数据库管理
    part2_database_management(cleaned_df)

    # 3. 执行 Part 3: 神经网络
    part3_neural_network()

    print("\n--- 所有任务执行完毕 ---")


if __name__ == '__main__':
    main()



# In[ ]:




