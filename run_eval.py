
import torch
import numpy as np
from core.data_utils.load import load_data
from core.LMs.utils import get_task_description, get_detailed_instruct, get_evaluator


dataset_name = 'ogbn-arxiv'

data, num_classes, text = load_data(
    dataset=dataset_name, use_text=True, use_gpt=False, seed=1)

num_nodes = data.y.size(0)

pred_path = f'prt_lm_finetuned/{dataset_name}/Salesforce/SFR-Embedding-Mistral-seed1.pred'

pred = torch.from_numpy(np.array(
    np.memmap(pred_path, mode='r',
                dtype=np.float16,
                shape=(num_nodes, num_classes)))
).to(torch.float32)


labels = data.y.numpy()


eval = get_evaluator(dataset_name, pred, labels)


train_acc = eval(data.train_mask)
val_acc = eval(data.val_mask)
test_acc = eval(data.test_mask)

print(
    f'[LM] TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')

