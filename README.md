# 门店特征重要性分析系统

这是一个基于机器学习的门店特征重要性分析系统，用于分析影响门店日营业额的各种因素。项目使用多种机器学习方法（BP神经网络、随机森林等）来评估不同特征对营业额的影响程度。

## 功能特点

- **多模型分析**：使用BP神经网络、随机森林等多种机器学习方法进行特征重要性分析
- **全面的特征评估**：包含排列重要性、SHAP值等多种特征重要性评估方法
- **可视化分析**：提供直观的特征重要性可视化图表
- **模型稳定性分析**：包含权重稳定性分析，确保结果的可靠性
- **中文支持**：完整支持中文显示和输出

## 数据说明

数据集包含以下主要特征：

- 门店基本信息（名称、所在行政区）
- 500米范围内的环境特征（住宅人口、商业场所客流量、青少年人数等）
- 基础设施特征（地铁出口、公交站、商场、学校数量等）
- 目标变量：日营业额

## 安装说明

1. 克隆项目到本地：

```bash
git clone [项目地址]
cd [项目目录]
```

1. 安装依赖包：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：
   - 将数据文件 `data.csv` 放在项目根目录下
   - 确保数据格式符合预期（参考数据说明部分）

2. 运行分析：

```bash
python feature_importance_analysis.py
```

1. 查看结果：
   - 分析结果将保存在 `results` 目录下
   - 包含特征重要性图表和详细的分析报告

## 项目结构

``` bash
项目根目录/
├── feature_importance_analysis.py  # 主分析脚本
├── data.csv                        # 原始数据
├── results/                        # 结果输出目录
│   └── pictures/                   # 可视化图表
└── README.md                       # 项目说明文档
```

## 主要功能模块

- `load_and_preprocess_data()`: 数据加载和预处理
- `train_bp_network()`: BP神经网络训练
- `train_random_forest()`: 随机森林模型训练
- `calculate_permutation_importance()`: 计算排列重要性
- `calculate_shap_values()`: 计算SHAP值
- `visualize_feature_importance()`: 特征重要性可视化
- `analyze_weights_stability()`: 分析权重稳定性

## 依赖包

- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- shap

## 注意事项

1. 确保数据文件格式正确
2. 建议使用Python 3.7或更高版本
3. 首次运行时可能需要安装额外的依赖包
4. 确保有足够的磁盘空间存储分析结果

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。在提交代码前，请确保：

1. 代码符合PEP 8规范
2. 添加适当的注释
3. 更新相关文档
4. 通过所有测试

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件至[hy20051123@gmail.com]
