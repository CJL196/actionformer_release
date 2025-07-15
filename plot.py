import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_segmentation(ground_truth, prediction, class_labels, save_path):
    """
    绘制时序动作分割的结果比较图，并添加图例。
    
    参数:
    - ground_truth: list[int]，真实的动作类别列表
    - prediction: list[int]，模型预测的动作类别列表
    - class_labels: list[str]，类别编号到类别名称的映射
    - save_path: str，保存图片的路径
    """
    assert len(ground_truth) == len(prediction), "两个列表长度必须相同"
    
    # 使用自定义颜色映射
    num_classes = len(class_labels)
    colors = plt.get_cmap('tab20', num_classes)
    
    # 将数据转换为numpy数组
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)
    
    # 绘图
    fig, ax = plt.subplots(2, 1, figsize=(12, 2.5), sharex=True)
    
    # 绘制Ground Truth
    ax[0].imshow(ground_truth.reshape(1, -1), aspect='auto', cmap=colors, vmin=0, vmax=num_classes-1)
    ax[0].set_title('Ground Truth')
    ax[0].axis('off')
    
    # 绘制Prediction
    ax[1].imshow(prediction.reshape(1, -1), aspect='auto', cmap=colors, vmin=0, vmax=num_classes-1)
    ax[1].set_title('Predict')
    ax[1].axis('off')
    
    # 添加图例
    legend_elements = [Patch(facecolor=colors(i), label=f'{i}: {class_labels[i]}') 
                       for i in range(num_classes)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=num_classes, fontsize=8)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    
    # 保存图片
    plt.savefig(save_path, dpi=300)
    plt.close()
