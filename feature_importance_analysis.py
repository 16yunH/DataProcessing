import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import seaborn as sns
import os
import shap
import warnings
import matplotlib
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用SimHei
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'SimHei'

# 创建一个列表存储所有图表，以便最后一起显示
all_figures = []

def load_and_preprocess_data(data_file='data.csv'):
    """加载并预处理数据"""
    # 首先检查文件是否存在
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"文件 {data_file} 不存在")
    
    print(f"开始加载数据文件：{data_file}")
    
    # 定义预期的列名
    expected_columns = [
        '门店',
        '所在行政区',
        '500米内住宅区常住人口数/人',
        '500米内商业场所日均客流量/人',
        '500米内青少年人数/人',
        '500米内餐饮消费水平/元',
        '500米内地铁出口数量/个',
        '500米内公交站数量/个',
        '500米内商场数量/个',
        '500米内学校数量/个',
        '500米内住宅区数量/个',
        '一公里内其他加盟店数量/个',
        '日营业额'
    ]
    
    # 定义用于分析的特征列（第3列到第12列）
    feature_columns = expected_columns[2:-1]  # 不包括'门店'、'所在行政区'和'日营业额'
    
    try:
        # 尝试直接读取数据，跳过可能损坏的行
        data = pd.read_csv(data_file, encoding='gbk', on_bad_lines='skip')
        
        # 检查并修复列名
        if len(data.columns) == len(expected_columns):
            data.columns = expected_columns
            print("成功设置预定义的列名")
        else:
            print(f"警告：列数不匹配。预期{len(expected_columns)}列，实际{len(data.columns)}列")
            print(f"实际列名: {data.columns.tolist()}")
            return None, None, None
        
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None, None
    
    # 检查数据是否读取成功
    if data.empty:
        print("警告：数据读取后为空")
        return None, None, None
    
    print(f"数据维度：{data.shape}")
    print(f"数据列名：{data.columns.tolist()}")
    
    # 保存原始数据记录数，用于后续比较和报告
    original_rows = data.shape[0]
    print(f"原始数据行数：{original_rows}")
    
    # 创建一个副本进行处理
    data_for_analysis = data.copy()
    
    # 将门店列设为索引（用于参考，但不参与分析）
    if '门店' in data_for_analysis.columns:
        print("将'门店'列设置为索引")
        data_for_analysis = data_for_analysis.set_index('门店')
    
    # 提取目标变量
    target_column = '日营业额'
    if target_column not in data_for_analysis.columns:
        print(f"错误：找不到目标变量列 '{target_column}'")
        return None, None, None
    
    y = data_for_analysis[target_column].copy()
    
    # 只选择用于分析的特征列
    X = data_for_analysis[feature_columns].copy()
    
    # 尝试将特征列转为数值类型
    for col in X.columns:
        try:
            # 清理数据中的空格
            X[col] = X[col].str.strip()
            # 使用coerce参数将无法转换的值设为NaN
            X[col] = pd.to_numeric(X[col], errors='coerce')
            print(f"将列 '{col}' 转换为数值类型")
        except Exception as e:
            print(f"将列 '{col}' 转换为数值时出错: {e}")
    
    # 同样处理目标变量
    try:
        y = y.str.strip()
        y = pd.to_numeric(y, errors='coerce')
        print(f"将目标变量 '{target_column}' 转换为数值类型")
    except Exception as e:
        print(f"将目标变量转换为数值时出错: {e}")
    
    # 检查是否有完全为空的行（所有特征都是NaN）
    empty_rows = X.isnull().all(axis=1)
    print(f"完全为空的行数: {empty_rows.sum()}")
    
    # 移除包含缺失值过多的列（例如超过80%的值缺失）
    threshold = len(X) * 0.2  # 要求至少20%的数据是有效的
    columns_before = X.columns.tolist()
    X = X.dropna(axis=1, thresh=threshold)
    columns_after = X.columns.tolist()
    dropped_columns = set(columns_before) - set(columns_after)
    
    if dropped_columns:
        print(f"已移除以下缺失值过多的列: {dropped_columns}")
    
    # 检查并处理存在缺失值的行
    rows_with_missing = X.isnull().any(axis=1) | y.isnull()
    print(f"特征或目标变量中存在缺失值的行数: {rows_with_missing.sum()}")
    
    # 同时移除X和y中的缺失值行
    complete_index = ~rows_with_missing
    X_complete = X[complete_index]
    y_complete = y[complete_index]
    
    print(f"移除缺失值后，用于分析的完整行数: {X_complete.shape[0]}")
    
    if X_complete.shape[0] < 10:
        print("警告：有效数据太少，可能影响分析质量。")
        if X_complete.shape[0] == 0:
            print("错误：没有完整的数据行可以用于分析！")
            return None, None, None
    
    # 保存真实的特征名称
    feature_names = X_complete.columns.tolist()
    
    print(f"\n最终选择的特征 ({len(feature_names)}个):")
    for i, feature in enumerate(feature_names, 1):
        print(f"{i}. {feature}")
    
    print(f"\n目标变量: {target_column}")
    print(f"最终用于分析的数据维度: {X_complete.shape}")
    print(f"目标变量的样本数量: {len(y_complete)}")
    
    # 验证X和y的样本数量是否一致
    if len(X_complete) != len(y_complete):
        print("错误：特征矩阵和目标变量的样本数量不一致！")
        return None, None, None
    
    # 统计数据
    print("\n数据统计概览:")
    print(f"原始行数: {original_rows}")
    print(f"最终用于分析的行数: {X_complete.shape[0]}")
    print(f"数据利用率: {X_complete.shape[0]/original_rows*100:.2f}%")
    
    return X_complete, y_complete, feature_names

def generate_sample_data(n_samples=100, n_features=5, random_state=42):
    """生成示例数据用于演示"""
    print("\n由于无法加载有效的数据文件，生成示例数据用于演示...")
    np.random.seed(random_state)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 定义真实的特征权重
    weights = np.array([0.5, 0.8, 0.1, 0.3, 0.9])
    
    # 生成目标变量
    y = np.dot(X, weights) + np.random.randn(n_samples) * 0.1
    
    # 创建特征名称
    feature_names = [f'特征{i+1}' for i in range(n_features)]
    
    # 转换为DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y, feature_names

def train_bp_network(X, y, feature_names, hidden_layers=(10, 5), random_state=42):
    """训练BP神经网络并分析特征重要性"""
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建并训练BP神经网络
    print("\n开始训练BP神经网络...")
    bp_model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=random_state,
        verbose=True
    )
    
    bp_model.fit(X_train_scaled, y_train)
    
    # 评估模型性能
    y_pred = bp_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n模型评估:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"决定系数(R²): {r2:.4f}")
    
    # 获取输入层权重
    input_weights = bp_model.coefs_[0]  # 形状为 [n_features, n_neurons_first_hidden_layer]
    
    # 计算每个特征的权重绝对值的平均值
    feature_importance = np.mean(np.abs(input_weights), axis=1)
    
    # 归一化特征重要性，使总和为1
    normalized_importance = feature_importance / np.sum(feature_importance)
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': normalized_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n特征重要性 (BP网络输入层权重绝对值均值):")
    print(importance_df)
    
    return bp_model, importance_df, scaler, X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test, feature_names, random_state=42):
    """训练随机森林模型并分析特征重要性"""
    print("\n开始训练随机森林模型...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state
    )
    
    rf_model.fit(X_train, y_train)
    
    # 评估模型性能
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n随机森林模型评估:")
    print(f"均方误差(MSE): {mse:.4f}")
    print(f"决定系数(R²): {r2:.4f}")
    
    # 获取特征重要性
    rf_importance = rf_model.feature_importances_
    
    # 创建特征重要性数据框
    rf_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n随机森林特征重要性:")
    print(rf_importance_df)
    
    return rf_model, rf_importance_df

def calculate_permutation_importance(model, X_test, y_test, feature_names, random_state=42):
    """计算排列重要性"""
    print("\n计算排列重要性...")
    
    # 计算排列重要性
    perm_importance = permutation_importance(
        model, X_test, y_test, 
        n_repeats=10, 
        random_state=random_state
    )
    
    # 创建特征重要性数据框
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("\n排列重要性:")
    print(perm_importance_df)
    
    return perm_importance_df

def calculate_shap_values(model, X_train, X_test, feature_names, model_type='rf'):
    """计算SHAP值"""
    print(f"\n计算{model_type.upper()}模型的SHAP值...")
    
    # 根据模型类型创建SHAP解释器
    if model_type == 'rf':
        # 对于随机森林，使用Tree解释器
        explainer = shap.TreeExplainer(model)
        # 使用训练集的子样本加速计算
        shap_values = explainer.shap_values(X_test[:100] if len(X_test) > 100 else X_test)
    else:
        # 对于神经网络，使用KernelExplainer
        background = shap.kmeans(X_train, 10)  # 使用K-means摘要作为背景数据
        explainer = shap.KernelExplainer(model.predict, background)
        # 使用测试集的子样本加速计算
        shap_values = explainer.shap_values(X_test[:50] if len(X_test) > 50 else X_test)
    
    # 计算每个特征的全局重要性（绝对SHAP值的平均值）
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # 归一化特征重要性
    normalized_importance = feature_importance / np.sum(feature_importance)
    
    # 创建特征重要性数据框
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': normalized_importance
    }).sort_values('Importance', ascending=False)
    
    print(f"\nSHAP值特征重要性 ({model_type.upper()}模型):")
    print(shap_importance_df)
    
    return shap_importance_df, explainer, shap_values

def visualize_feature_importance(importance_df, title='BP神经网络特征重要性', filename='results/pictures/bp_feature_importance.png'):
    """可视化特征重要性"""
    fig = plt.figure(figsize=(12, 8))
    
    # 按重要性降序排列
    importance_df = importance_df.sort_values('Importance')
    
    # 创建水平条形图
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    
    plt.title(title, fontsize=16)
    plt.xlabel('重要性分数', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n特征重要性图已保存为 '{filename}'")
    
    # 添加到图表列表
    all_figures.append((fig, title))
    
    return fig

def visualize_shap_summary(explainer, shap_values, X_test, feature_names, model_type='rf'):
    """可视化SHAP摘要图"""
    fig = plt.figure(figsize=(12, 8))
    
    # 创建特征名称的映射
    feature_map = {i: name for i, name in enumerate(feature_names)}
    
    # 创建SHAP摘要图
    if model_type == 'rf':
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    
    plt.title(f"SHAP值摘要图 ({model_type.upper()}模型)", fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    filename = f'results/pictures/shap_summary_{model_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSHAP摘要图已保存为 '{filename}'")
    
    # 添加到图表列表
    all_figures.append((fig, f"SHAP值摘要图 ({model_type.upper()}模型)"))
    
    return fig

def compare_feature_importance(bp_importance, rf_importance, perm_importance, shap_rf_importance, feature_names):
    """比较不同方法的特征重要性"""
    print("\n比较不同方法的特征重要性...")
    
    # 合并所有方法的特征重要性
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'BP_NN_Weights': bp_importance['Importance'].values,
        'RandomForest': rf_importance['Importance'].values,
        'Permutation': perm_importance['Importance'].values,
        'SHAP_RF': shap_rf_importance['Importance'].values
    })
    
    # 计算每个特征在各种方法中的平均排名
    methods = ['BP_NN_Weights', 'RandomForest', 'Permutation', 'SHAP_RF']
    for method in methods:
        comparison_df[f'{method}_Rank'] = comparison_df[method].rank(ascending=False)
    
    comparison_df['Average_Rank'] = comparison_df[[f'{method}_Rank' for method in methods]].mean(axis=1)
    comparison_df = comparison_df.sort_values('Average_Rank')
    
    print("\n特征重要性比较结果:")
    print(comparison_df)
    
    # 可视化比较结果
    fig = plt.figure(figsize=(14, 10))
    
    # 转换数据框为长格式，便于绘图
    plot_df = pd.melt(
        comparison_df, 
        id_vars=['Feature', 'Average_Rank'], 
        value_vars=methods,
        var_name='Method', 
        value_name='Importance'
    )
    
    # 按平均排名排序
    features_sorted = comparison_df.sort_values('Average_Rank')['Feature'].tolist()
    plot_df['Feature'] = pd.Categorical(plot_df['Feature'], categories=features_sorted, ordered=True)
    
    # 创建分组条形图
    sns.barplot(x='Importance', y='Feature', hue='Method', data=plot_df, palette='Set2')
    
    plt.title("不同方法的特征重要性比较", fontsize=16)
    plt.xlabel('重要性分数 (归一化)', fontsize=12)
    plt.ylabel('特征 (按平均排名排序)', fontsize=12)
    plt.legend(title='评估方法', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('results/pictures/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("\n特征重要性比较图已保存为 'results/pictures/feature_importance_comparison.png'")
    
    # 添加到图表列表
    all_figures.append((fig, "不同方法的特征重要性比较"))
    
    return comparison_df

def analyze_weights_stability(X, y, feature_names, n_runs=5):
    """分析权重稳定性"""
    print("\n分析权重稳定性，进行多次训练...")
    importance_scores = []
    
    for i in range(n_runs):
        print(f"\n运行 {i+1}/{n_runs}")
        _, importance_df, *_ = train_bp_network(X, y, feature_names, random_state=i)
        importance_scores.append(importance_df['Importance'].values)
    
    # 将所有运行的重要性分数转换为数组
    importance_array = np.array(importance_scores)
    
    # 计算平均重要性和标准差
    mean_importance = np.mean(importance_array, axis=0)
    std_importance = np.std(importance_array, axis=0)
    
    # 创建结果数据框
    stability_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': mean_importance,
        'Std_Importance': std_importance,
        'Coefficient_of_Variation': std_importance / mean_importance
    }).sort_values('Mean_Importance', ascending=False)
    
    print("\n权重稳定性分析结果:")
    print(stability_df)
    
    # 可视化稳定性分析
    fig = plt.figure(figsize=(12, 8))
    
    # 按平均重要性降序排列
    stability_df = stability_df.sort_values('Mean_Importance')
    
    # 创建水平条形图
    plt.barh(stability_df['Feature'], stability_df['Mean_Importance'], 
             xerr=stability_df['Std_Importance'], capsize=5, color='skyblue', alpha=0.7)
    
    plt.title("BP神经网络特征重要性及其稳定性", fontsize=16)
    plt.xlabel('平均重要性分数 (误差棒表示标准差)', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('results/pictures/bp_feature_importance_stability.png', dpi=300, bbox_inches='tight')
    print("\n权重稳定性分析图已保存为 'results/pictures/bp_feature_importance_stability.png'")
    
    # 添加到图表列表
    all_figures.append((fig, "BP神经网络特征重要性及其稳定性"))
    
    return stability_df

def show_all_figures():
    """在分析完成后一次性显示所有图表"""
    print("\n显示所有可视化结果...")
    
    if not all_figures:
        print("没有图表需要显示")
        return
    
    num_figures = len(all_figures)
    cols = min(2, num_figures)  # 最多2列
    rows = (num_figures + cols - 1) // cols  # 计算需要的行数
    
    plt.figure(figsize=(15, 6 * rows))
    
    for i, (fig, title) in enumerate(all_figures):
        plt.figure(fig.number)
        plt.tight_layout()
    
    plt.show()

def main():
    """主函数"""
    print("=== 基于多种方法的特征重要性分析 ===")
    
    # 检查可用的数据文件
    available_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"可用的CSV文件: {available_files}")
    
    # 尝试读取数据
    X, y, feature_names = None, None, None
    
    # 首先尝试colddrink_store_data.csv
    try:
        if 'colddrink_store_data.csv' in available_files:
            X, y, feature_names = load_and_preprocess_data('colddrink_store_data.csv')
        elif 'data.csv' in available_files:
            X, y, feature_names = load_and_preprocess_data('data.csv')
        else:
            raise FileNotFoundError("找不到合适的数据文件")
    except Exception as e:
        print(f"加载数据文件失败: {e}")
        # 如果无法加载数据，生成示例数据
        X, y, feature_names = generate_sample_data()
    
    # 创建results目录以及pictures和models子目录保存结果
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/pictures', exist_ok=True)
    os.makedirs('results/models', exist_ok=True)
    
    print("\n=== 第1阶段：模型训练 ===")
    # 1. 首先训练所有模型
    print("\n训练BP神经网络模型...")
    bp_model, bp_importance_df, scaler, X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test = train_bp_network(X, y, feature_names)
    
    print("\n训练随机森林模型...")
    rf_model, rf_importance_df = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    
    # 保存模型
    import pickle
    print("\n保存训练好的模型...")
    with open('results/models/bp_model.pkl', 'wb') as f:
        pickle.dump(bp_model, f)
    with open('results/models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('results/models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n=== 第2阶段：特征重要性分析 ===")
    # 2. 基于训练好的模型进行特征重要性分析
    # 2.1 可视化BP网络特征重要性
    print("\n分析BP神经网络特征重要性...")
    visualize_feature_importance(bp_importance_df, 
                               title='BP神经网络权重特征重要性', 
                               filename='results/pictures/bp_weights_importance.png')
    
    # 2.2 可视化随机森林特征重要性
    print("\n分析随机森林特征重要性...")
    visualize_feature_importance(rf_importance_df, 
                               title='随机森林特征重要性', 
                               filename='results/pictures/random_forest_importance.png')
    
    # 2.3 分析排列重要性 (使用随机森林模型)
    print("\n计算基于随机森林的排列重要性...")
    perm_importance_df = calculate_permutation_importance(rf_model, X_test, y_test, feature_names)
    visualize_feature_importance(perm_importance_df, 
                               title='排列重要性', 
                               filename='results/pictures/permutation_importance.png')
    
    # 2.4 计算SHAP值 (对随机森林模型)
    shap_rf_importance_df = None
    try:
        print("\n计算基于随机森林的SHAP值...")
        shap_rf_importance_df, rf_explainer, rf_shap_values = calculate_shap_values(
            rf_model, X_train, X_test, feature_names, model_type='rf'
        )
        visualize_feature_importance(shap_rf_importance_df, 
                                   title='SHAP值特征重要性 (随机森林)', 
                                   filename='results/pictures/shap_rf_importance.png')
        visualize_shap_summary(rf_explainer, rf_shap_values, X_test, feature_names, model_type='rf')
    except Exception as e:
        print(f"计算SHAP值时出错: {e}")
        print("跳过SHAP值分析...")
        shap_rf_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.zeros(len(feature_names))
        })
    
    print("\n=== 第3阶段：对比分析 ===")
    # 3. 比较不同方法的结果
    print("\n比较所有特征重要性方法...")
    comparison_df = compare_feature_importance(
        bp_importance_df, rf_importance_df, perm_importance_df, shap_rf_importance_df, feature_names
    )
    
    # 4. 分析BP权重稳定性（这是一个单独的分析，会重新训练多个模型）
    print("\n进行BP网络权重稳定性分析（将训练多个模型）...")
    stability_df = analyze_weights_stability(X, y, feature_names)
    
    # 保存所有结果
    print("\n保存所有分析结果...")
    bp_importance_df.to_csv('results/bp_weights_importance.csv', index=False)
    rf_importance_df.to_csv('results/random_forest_importance.csv', index=False)
    perm_importance_df.to_csv('results/permutation_importance.csv', index=False)
    if shap_rf_importance_df is not None:
        shap_rf_importance_df.to_csv('results/shap_rf_importance.csv', index=False)
    comparison_df.to_csv('results/feature_importance_comparison.csv', index=False)
    stability_df.to_csv('results/bp_weights_stability.csv', index=False)
    
    # 在所有分析完成后显示所有图表
    show_all_figures()
    
    print("\n分析完成！所有图片已保存到 'results/pictures' 目录，模型已保存到 'results/models' 目录。")
    
    print("\n=== 第4阶段：综合结论 ===")
    # 打印综合结论
    print("\n===== 综合分析结论 =====")
    print("按照平均排名，最重要的特征是:")
    for i, (feature, rank) in enumerate(zip(comparison_df['Feature'].values[:3], comparison_df['Average_Rank'].values[:3])):
        print(f"{i+1}. {feature} (平均排名: {rank:.2f})")
    
    print("\n特征重要性一致性分析:")
    methods = ['BP_NN_Weights', 'RandomForest', 'Permutation', 'SHAP_RF']
    top_features_by_method = {}
    for method in methods:
        top_features = comparison_df.sort_values(method, ascending=False)['Feature'].values[:3].tolist()
        top_features_by_method[method] = top_features
        print(f"{method} 排名前三特征: {', '.join(top_features)}")
    
    # 分析不同方法结果的一致性
    all_top_features = []
    for features in top_features_by_method.values():
        all_top_features.extend(features)
    
    feature_counts = {}
    for feature in all_top_features:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    consensus_features = [feature for feature, count in feature_counts.items() if count >= 2]
    print(f"\n共识特征 (至少两种方法认为重要): {', '.join(consensus_features)}")

if __name__ == "__main__":
    main()