import numpy as np

# 数据
psc = [0, 0, 1, 2, 2, 3, 3, 4, 2, 3, 3, 1, 3, 2, 3, 4, 2, 3, 1, 4, 2, 3, 3, 4, 4]
teacher_score = [0, 0, 1, 2, 2, 3, 3, 3, 1, 3, 3, 1, 3, 2, 4, 4, 2, 3, 0, 4, 4, 2, 2, 4, 4]

# 样本数量
n = len(psc)

# 计算均值
mean_psc = sum(psc) / n
mean_teacher_score = sum(teacher_score) / n

# 计算皮尔逊系数的分子和分母
numerator = 0
denominator_psc = 0
denominator_teacher = 0

for i in range(n):
    # 分子：(x_i - mean_x)(y_i - mean_y)
    numerator += (psc[i] - mean_psc) * (teacher_score[i] - mean_teacher_score)
    # 分母：(x_i - mean_x)^2 和 (y_i - mean_y)^2
    denominator_psc += (psc[i] - mean_psc) ** 2
    denominator_teacher += (teacher_score[i] - mean_teacher_score) ** 2

# 分母：sqrt(Σ(x_i - mean_x)^2) * sqrt(Σ(y_i - mean_y)^2)
denominator = (denominator_psc ** 0.5) * (denominator_teacher ** 0.5)

# 计算皮尔逊系数
r = numerator / denominator

# 输出结果
print(f"均值 Psc: {mean_psc:.2f}")
print(f"均值 教员评分: {mean_teacher_score:.2f}")
print(f"皮尔逊相关系数 r: {r:.4f}")

# 使用 numpy 验证
r_numpy = np.corrcoef(psc, teacher_score)[0, 1]
print(f"使用 numpy 计算的皮尔逊系数: {r_numpy:.4f}")