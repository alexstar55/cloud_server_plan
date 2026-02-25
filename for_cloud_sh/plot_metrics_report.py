import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def generate_report(csv_path, output_dir):
    """
    根据包含 [distance, class, iou, is_occluded] 的 CSV 文件生成切片报表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] 找不到文件: {csv_path}")
        return
    
    # 确保必要的列存在
    required_cols = ['distance', 'class', 'iou', 'is_occluded']
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] CSV 文件缺少必要的列: {col}")
            return

    # 1. 距离分桶 mAP 柱状图 (假设 iou > 0.5 算作 True Positive，这里简化计算)
    # 实际应用中，mAP 的计算更复杂，这里用 Recall 或简单的 TP 率代替展示趋势
    df['distance_bin'] = pd.cut(df['distance'], bins=[0, 30, 50, 80, 100], labels=['0-30m', '30-50m', '50-80m', '80-100m'])
    df['is_tp'] = df['iou'] > 0.5
    
    dist_metrics = df.groupby('distance_bin')['is_tp'].mean().reset_index()
    dist_metrics.rename(columns={'is_tp': 'accuracy_proxy'}, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='distance_bin', y='accuracy_proxy', data=dist_metrics, palette='viridis')
    plt.title('Detection Accuracy Proxy by Distance')
    plt.xlabel('Distance Range')
    plt.ylabel('Accuracy Proxy (IoU > 0.5)')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, 'distance_metrics.png'))
    plt.close()
    print(f"[INFO] 已生成距离分桶报表: {os.path.join(output_dir, 'distance_metrics.png')}")

    # 2. bus 类召回率曲线 (按距离)
    bus_df = df[df['class'] == 'bus']
    if not bus_df.empty:
        bus_dist_metrics = bus_df.groupby('distance_bin')['is_tp'].mean().reset_index()
        bus_dist_metrics.rename(columns={'is_tp': 'recall_proxy'}, inplace=True)

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='distance_bin', y='recall_proxy', data=bus_dist_metrics, marker='o', color='r')
        plt.title('Bus Recall Proxy by Distance')
        plt.xlabel('Distance Range')
        plt.ylabel('Recall Proxy (IoU > 0.5)')
        plt.ylim(0, 1.0)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'bus_recall_by_distance.png'))
        plt.close()
        print(f"[INFO] 已生成 bus 类距离召回率报表: {os.path.join(output_dir, 'bus_recall_by_distance.png')}")
    else:
        print("[WARN] 数据中没有 'bus' 类别，跳过生成 bus 报表。")

    # 3. 遮挡情况分析
    occ_metrics = df.groupby('is_occluded')['is_tp'].mean().reset_index()
    occ_metrics.rename(columns={'is_tp': 'accuracy_proxy'}, inplace=True)
    occ_metrics['is_occluded'] = occ_metrics['is_occluded'].map({True: 'Occluded', False: 'Not Occluded'})

    plt.figure(figsize=(8, 6))
    sns.barplot(x='is_occluded', y='accuracy_proxy', data=occ_metrics, palette='Set2')
    plt.title('Detection Accuracy Proxy by Occlusion')
    plt.xlabel('Occlusion Status')
    plt.ylabel('Accuracy Proxy (IoU > 0.5)')
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(output_dir, 'occlusion_metrics.png'))
    plt.close()
    print(f"[INFO] 已生成遮挡情况报表: {os.path.join(output_dir, 'occlusion_metrics.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metrics report from CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--out", type=str, default="./report_output", help="Directory to save the report images.")
    args = parser.parse_args()
    
    generate_report(args.csv, args.out)
