import os
import json
import glob
import numpy as np
import pickle

def time_str_to_seconds(time_str):
    """
    将 '0-22,0-25' 形式的字符串转为 [22.0, 25.0] 秒
    """
    start, end = time_str.split(',')
    m1, s1 = map(int, start.split('-'))
    m2, s2 = map(int, end.split('-'))
    return [m1 * 60 + s1, m2 * 60 + s2]

def main():
    annotation_dir = 'data/FineFS/data/annotation'
    output_path = 'data/FineFS/actionformer.json'
    feat_root = './data/FineFS/process/swin_b_skipping1_merge32'
    version = 'FineFS-30fps'
    fps = 25.0
    database = {}
    label2id = {}
    next_label_id = 0

    indice_sp_path = 'data/FineFS/myindice/indices_sp.pkl'
    indice_fs_path = 'data/FineFS/myindice/indices_fs.pkl'

    with open(indice_sp_path, 'rb') as f:
        indices_sp = pickle.load(f)
    with open(indice_fs_path, 'rb') as f:
        indices_fs = pickle.load(f)

    train_indices = set(indices_sp['train'] + indices_fs['train'])
    test_indices = set(indices_sp['test'] + indices_fs['test'])

    for ann_path in glob.glob(os.path.join(annotation_dir, '*.json')):
        video_id = os.path.splitext(os.path.basename(ann_path))[0]
        
        try:
            int_video_id = int(video_id)
        except ValueError:
            continue

        if int_video_id in train_indices:
            subset = 'train'
        elif int_video_id in test_indices:
            subset = 'test'
        else:
            continue

        feat_path = os.path.join(feat_root, video_id + '.mp4.npy')
        feat = np.load(feat_path)
        with open(ann_path, 'r', encoding='utf-8') as f:
            ann = json.load(f)
        executed = ann.get('executed_element', {})
        annotations = []
        for ele in executed.values():
            label = ele.get('coarse_class', 'Unknown')  # 由coarse_class决定
            time_str = ele.get('time', None)
            if not time_str:
                continue
            try:
                segment = time_str_to_seconds(time_str)
            except Exception:
                continue
            if label not in label2id:
                label2id[label] = next_label_id
                next_label_id += 1
            label_id = label2id[label]
            annotations.append({
                'label': label,
                'segment': segment,
                'label_id': label_id
            })
        database[video_id] = {
            'subset': subset,
            'duration': feat.shape[0] * 32 / fps,
            'fps': fps,
            'annotations': annotations
        }

    out_json = {
        'version': version,
        'database': database
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f"不同label总数: {len(label2id)}")
    print("label到label_id的映射:")
    for label, idx in label2id.items():
        print(f"  {label}: {idx}")

if __name__ == '__main__':
    main()
