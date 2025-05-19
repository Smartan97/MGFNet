import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
import os
import pickle

path_Extracted = '/root/Smartan/datasets/ISRUC_S1/S1_ExtractedChannels/'
path_RawData   = '/root/Smartan/datasets/ISRUC_S1/S1_RawData/'
path_output    = '/root/Smartan/datasets/ISRUC_S1/processed/'
channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1', 'X1', 'X2']
# 说明  'LOC_A2', 'ROC_A1'为EOG   'x1'为Chin EMG  'x2'为ECG (EKG)
def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s=='' or s=='\n':
                break
    return np.array(label[:-ignore])

# 创建十折数据
num_subjects = 100
k_folds = 10
subjects = np.arange(1, num_subjects + 1)
np.random.shuffle(subjects)  # 打乱被试顺序
data_folds = np.array_split(subjects, k_folds)

fold_data = []
fold_label = []
fold_len = []

for i, fold in enumerate(data_folds):
    fold_psg = []
    fold_lab = []
    fold_lengths = []
    print(f'Processing fold {i+1}')
    
    for sub in fold:
        print('Read subject', sub)
        label = read_label(path_RawData, sub)
        psg = read_psg(path_Extracted, sub, channels)
        assert len(label) == len(psg)

        label[label == 5] = 4  # make 4 correspond to REM
        fold_lab.append(np.eye(5)[label])  # 转为one-hot编码
        fold_psg.append(psg)
        fold_lengths.append(len(label))
    
    fold_data.append(fold_psg)
    fold_label.append(fold_lab)
    fold_len.append(fold_lengths)

print('Preprocess over.')

os.makedirs(path_output, exist_ok=True)  # 如果目录不存在就创建
with open(os.path.join(path_output, 'ISRUC_S1.pkl'), 'wb') as f:
    pickle.dump({'Fold_data': fold_data, 'Fold_label': fold_label, 'Fold_len': fold_len}, f)
    
print('Saved to', path.join(path_output, 'ISRUC_S1.pkl'))



