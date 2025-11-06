# -*- coding: utf-8 -*-                                 # 指定源码文件的字符编码，避免中文注释在某些环境报错
# Surname_Name_StudentID                                # 交作业时把这里替换成你的姓名/学号

import pandas as pd                                     # 导入 pandas，做数据读取、清洗、汇总
import matplotlib.pyplot as plt                         # 导入 matplotlib，用于画图
import sqlite3                                          # 导入 sqlite3，用于创建和操作本地 SQLite 数据库
import torch                                            # 导入 PyTorch，用于构建和训练神经网络
from torch import nn                                    # 从 torch 中导入 nn 模块，定义网络结构
import numpy as np                                      # 导入 numpy，生成合成数据等
import os                             # 操作系统接口

# 临时允许重复 OpenMP 运行时继续执行（长期不推荐）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("\n======================")                        # 下面三行只是美化控制台输出
print(" TASK 1: DATA ANALYSIS")
print("======================\n")

# ---- Task 1a: Load Data ----
df = pd.read_csv("cyber_incidents.csv")                 # 读取同目录下的 CSV 数据为 DataFrame
print("▶ First 5 Rows:")                                # 打印提示信息
print(df.head(), "\n")                                  # 打印前 5 行，快速预览数据
print(f"▶ DataFrame Shape: {df.shape}\n")               # 打印 DataFrame 形状 (行数, 列数)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")# 将 Date 列转为日期类型；无法解析的置为 NaT
print(f"▶ Date dtype after conversion: {df['Date'].dtype}\n")  # 确认转换后的数据类型

# Handle missing values
df["Count"] = df["Count"].fillna(0)                     # Count 列缺失用 0 填充（计数型）
df = df.dropna(subset=["Date"]).reset_index(drop=True)  # 丢弃 Date 为 NaT 的行；重置索引保证连续

# Clean Incident_Type
df["Incident_Type"] = (                                 # 清洗事件类型字符串
    df["Incident_Type"].astype(str)                     # 统一转为字符串
                     .str.strip()                       # 去前后空格
                     .str.upper()                       # 转大写，规避 PHISHing/Phishing 等大小写不一
)
print("▶ Unique Incident Types:")                       # 打印提示
print(df["Incident_Type"].unique(), "\n")               # 查看清洗后的去重类型值

print("----- Task 1b: Visualisation Output -----\n")    # 分节标题

# Total incidents by type
totals_by_type = (                                      # 计算每个事件类型的总次数
    df.groupby("Incident_Type")["Count"]
      .sum()
      .sort_values(ascending=False)
)
print("▶ Total incidents per incident type:")           # 打印提示
print(totals_by_type, "\n")                             # 输出各类型总数

plt.figure()                                            # 创建一个新图形
totals_by_type.plot(kind="bar", color="steelblue")      # 画柱状图：X=类型，Y=总次数
plt.title("Total Incidents by Type")                    # 设置图标题
plt.xlabel("Incident Type")                             # X 轴标签
plt.ylabel("Total Count")                               # Y 轴标签
plt.tight_layout()                                      # 自适应边距避免标签被遮挡
plt.show()                                              # 显示图形（在脚本里会弹窗/在某些环境内嵌显示）

# ---- Monthly totals for 2025 (MUST show zeros for missing months) ----
df_2025 = df[df["Date"].dt.year == 2025].copy()         # 过滤出 2025 年的数据
df_2025["YearMonth"] = df_2025["Date"].dt.to_period("M").astype(str)  # 转为月份粒度的字符串如 '2025-03'

all_months = [f"2025-{m:02d}" for m in range(1, 13)]    # 明确列出 2025 年 12 个月（补齐缺失月份）

monthly_totals = (                                      # 汇总每月总数
    df_2025.groupby("YearMonth")["Count"]
          .sum()
          .reindex(all_months, fill_value=0)            # 用完整月份索引重建，缺的月填 0（关键：保证 0 也显示）
)

print("▶ Monthly Totals for 2025 (Including 0 Values):\n")  # 打印提示
print(monthly_totals, "\n")                             # 打印 12 个月每月总数（包含 0）

plt.figure(figsize=(8,4))                               # 新建图形并设置尺寸
plt.plot(                                               # 画折线图
    monthly_totals.index,                               # X 轴为月份字符串
    monthly_totals.values,                              # Y 轴为每月总数
    marker="o",                                         # 每个点用圆点标记
    markersize=8,                                       # 标记大小
    markerfacecolor="white",                            # 标记内填充白色，便于看清
    linestyle="-",                                      # 线型为实线
    linewidth=2,                                        # 线宽
    color="darkgreen"                                   # 折线颜色
)
plt.ylim(bottom=0)                                      # 强制 Y 轴从 0 开始，突出 0 值含义
for x, y in zip(monthly_totals.index, monthly_totals.values):  # 遍历每个点
    plt.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=9)  # 在点附近标数值（含 0）
plt.title("Monthly Incident Totals (2025)")             # 图标题
plt.xlabel("Month (YYYY-MM)")                           # X 轴标签
plt.ylabel("Total Incidents")                           # Y 轴标签
plt.grid(alpha=0.4, linestyle="--")                     # 添加虚线网格，增强可读性
plt.tight_layout()                                      # 调整边距
plt.show()                                              # 显示折线图

print("\n======================")                        # 分节装饰
print(" TASK 2: DATABASE WORK")
print("======================\n")

con = sqlite3.connect("incidentsDB.sqlite")             # 连接或创建 SQLite 数据库
cur = con.cursor()                                      # 获取游标对象，执行 SQL
# 建表：包含唯一性约束以避免重复插入
cur.execute("""                                         
CREATE TABLE IF NOT EXISTS incidents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Date TEXT,
    System TEXT,
    Incident_Type TEXT,
    Count INTEGER,
    UNIQUE(Date, System, Incident_Type)
);
""")

for row in df.itertuples(index=False):                   # 遍历 DataFrame 每一行记录 # 使用参数化插入，避免 SQL 注入/类型错误
    cur.execute("""                                     
    INSERT OR IGNORE INTO incidents (Date, System, Incident_Type, Count)
    VALUES (?, ?, ?, ?)
    """, (row.Date.strftime("%Y-%m-%d"),                # 将 datetime 格式化为 'YYYY-MM-DD'
          row.System,
          row.Incident_Type,
          int(row.Count)))                              # 确保 Count 是整数
con.commit()                                            # 提交事务，将插入写入数据库

print("✅ Data inserted into SQLite database (duplicates skipped).\n")  # 友好提示

print("▶ Total incidents in 2025:")                     # 打印提示
# 用 pandas 执行 SQL 并展示结果
print(pd.read_sql("""                                  
SELECT SUM(Count) AS Total_2025
FROM incidents
WHERE Date >= '2025-01-01' AND Date < '2026-01-01'
""", con), "\n")

print("▶ Top 3 Systems in 2025:")                       # 打印提示
# 查询 2025 年按系统汇总并取前 3 名
top3 = pd.read_sql("""                                  
SELECT System, SUM(Count) AS Total
FROM incidents
WHERE Date >= '2025-01-01' AND Date < '2026-01-01'
GROUP BY System
ORDER BY Total DESC
LIMIT 3
""", con)
print(top3, "\n")                                       # 打印前 3 系统及其总数

top3.to_csv("top3_systems.csv", index=False)            # 导出为 CSV 文件供上交/复查
print("💾 Exported to: top3_systems.csv\n")             # 输出保存路径提示

con.close()                                             # 关闭数据库连接，释放资源

print("======================")                            # 分节装饰
print(" TASK 3: NEURAL NETWORK")
print("======================\n")

torch.manual_seed(0)                                    # 固定随机种子，保证结果可复现
X = torch.randn(200, 4)                                 # 生成 200×4 的标准正态分布特征
y = (torch.rand(200, 1) > 0.5).float()                  # 生成 0/1 随机标签（二分类）

print(f"▶ X shape: {X.shape}")                          # 打印 X 的张量形状
print(f"▶ y shape: {y.shape}\n")                        # 打印 y 的张量形状

model = nn.Sequential(                                  # 定义前馈神经网络
    nn.Linear(4, 8),                                    # 全连接层：输入 4 维 -> 隐层 8 维
    nn.ReLU(),                                          # 激活函数 ReLU
    nn.Linear(8, 1),                                    # 隐层 8 维 -> 输出 1 维（概率）
    nn.Sigmoid()                                        # Sigmoid 将输出压到 0~1
)

print("▶ Model Architecture:\n")                        # 打印提示
print(model, "\n")                                      # 打印网络结构摘要

criterion = nn.BCELoss()                                # 使用二元交叉熵损失（与 Sigmoid 对应）
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)# 随机梯度下降优化器，学习率 0.05

losses = []                                             # 用于保存每个 epoch 的损失
print("▶ Training Progress (Epoch Loss):\n")            # 提示开始训练输出

for epoch in range(1, 51):                              # 训练 50 个 epoch（1..50）
    optimizer.zero_grad()                               # 每个 epoch 前先清零梯度
    output = model(X)                                   # 前向传播，得到预测概率
    loss = criterion(output, y)                         # 计算损失（预测 vs 真实标签）
    loss.backward()                                     # 反向传播，计算梯度
    optimizer.step()                                    # 根据梯度更新参数

    losses.append(loss.item())                          # 记录本次损失到列表中
    print(f"Epoch {epoch:02d}: Loss = {loss.item():.6f}")  # ✅ 打印每个 epoch 的损失

plt.figure()                                            # 训练后画损失曲线
plt.plot(losses, marker="o", color="purple")            # 折线图 + 圆点标记
plt.title("Neural Training Loss")                       # 图标题
plt.xlabel("Epoch")                                     # X 轴：训练轮次
plt.ylabel("Loss")                                      # Y 轴：损失
plt.grid(alpha=0.3)                                     # 添加网格提升可读性
plt.tight_layout()                                      # 调整边距
plt.show()                                              # 显示损失曲线

print("\n✅ Training complete.")                         # 训练完成提示
print(f"Final Loss: {losses[-1]:.6f}\n")                # 打印最终一次的损失
print("---- END OF SCRIPT ----")                        # 脚本结束标记
