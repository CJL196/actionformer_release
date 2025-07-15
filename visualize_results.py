import os
import json
import random
import numpy as np
from plot import plot_segmentation

def create_frame_level_representation(segments, num_frames, feat_stride, num_classes):
    """
    将分段数据转换为帧级别表示
    """
    frame_level_labels = np.zeros(num_frames, dtype=int)
    for seg in segments:
        start_time, end_time = seg['segment']
        label_id = seg['label_id'] if 'label_id' in seg else seg['label']

        # background is 0, so actions are 1-indexed in plot
        label_to_assign = label_id + 1
        
        # convert time to feature index
        start_idx = int(start_time / (feat_stride / 30.0)) # Assuming original FPS is 30
        end_idx = int(end_time / (feat_stride / 30.0))

        frame_level_labels[start_idx:end_idx] = label_to_assign
        
    return frame_level_labels.tolist()

def process_and_plot(video_ids, gt_data, pred_data, pred_keys_map, save_dir, class_labels_for_plot, feat_stride, num_classes):
    """
    Helper function to process a list of videos and save their plots.
    """
    for video_id in video_ids:
        print(f"Processing video: {video_id}")
        gt_info = gt_data[video_id]
        pred_key = pred_keys_map[video_id]
        pred_segments = pred_data[pred_key]
        
        # 获取视频总帧数 (feature length)
        # duration * fps / feat_stride
        num_frames = int(gt_info['duration'] * gt_info['fps'] / feat_stride)


        # 为GT和Prediction创建帧级别表示
        gt_frame_labels = create_frame_level_representation(gt_info['annotations'], num_frames, feat_stride, num_classes)
        pred_frame_labels = create_frame_level_representation(pred_segments, num_frames, feat_stride, num_classes)
        
        # 绘图
        save_path = os.path.join(save_dir, f'{pred_key}.png')
        plot_segmentation(gt_frame_labels, pred_frame_labels, class_labels_for_plot, save_path)


def main():
    # --- 配置 ---
    gt_json_path = 'data/FineFS/actionformer.json'
    pred_json_path = 'ckpt/finefs_swin_finefs/eval_results.json'
    base_save_dir = 'ckpt/finefs_swin_finefs/plots'
    train_save_dir = os.path.join(base_save_dir, 'train')
    test_save_dir = os.path.join(base_save_dir, 'test')
    num_train_samples = 20
    num_test_samples = 20
    # 从配置文件中获取的参数，如果找不到就用默认值
    feat_stride = 32 
    num_classes = 4 # jump, spin, sequence, step 

    # --- 准备工作 ---
    os.makedirs(train_save_dir, exist_ok=True)
    os.makedirs(test_save_dir, exist_ok=True)

    # 加载数据
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)['database']
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    # 分离训练集和测试集
    train_video_ids = [vid for vid, data in gt_data.items() if data['subset'] == 'train']
    test_video_ids = [vid for vid, data in gt_data.items() if data['subset'] == 'test']

    # 随机选择样本
    # 注意：eval_results.json中的key可能是'v_0'，而gt中是'0'
    pred_keys_map = {k.split('_')[-1]: k for k in pred_data.keys()}
    
    valid_train_ids = [vid for vid in train_video_ids if vid in pred_keys_map]
    valid_test_ids = [vid for vid in test_video_ids if vid in pred_keys_map]

    train_samples = random.sample(valid_train_ids, min(num_train_samples, len(valid_train_ids)))
    test_samples = random.sample(valid_test_ids, min(num_test_samples, len(valid_test_ids)))
    
    print(f"Total train videos to plot: {len(train_samples)}")
    print(f"Train samples: {train_samples}")
    print(f"Total test videos to plot: {len(test_samples)}")
    print(f"Test samples: {test_samples}")

    # --- 生成绘图 ---
    temp_labels = {0: 'jump', 1: 'spin', 2: 'sequence', 3: 'step'}
    class_labels_for_plot = ['background'] + [temp_labels[i] for i in range(num_classes)]
    
    # Process and plot train samples
    print("\n--- Processing Train Set ---")
    process_and_plot(train_samples, gt_data, pred_data, pred_keys_map, train_save_dir, class_labels_for_plot, feat_stride, num_classes)
    print(f"Train plots saved to {train_save_dir}")

    # Process and plot test samples
    print("\n--- Processing Test Set ---")
    process_and_plot(test_samples, gt_data, pred_data, pred_keys_map, test_save_dir, class_labels_for_plot, feat_stride, num_classes)
    print(f"Test plots saved to {test_save_dir}")


if __name__ == '__main__':
    main() 