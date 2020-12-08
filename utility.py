import torch

def calculate_mse(x1_list, x2_list):
	'''inputs as lists'''
	size = len(x1_list)
	total_error = 0
	for x1, x2 in zip(x1_list, x2_list):
		err = x2 - x1
		err2 = err ** 2
		total_error += err2
	mse = total_error/size
	return mse


def calculate_pcc(y_pred_useful, y):
	'''
	inputs as tensors
	'''
	size = y.size(0)
	vy = y - torch.mean(y)
	vyp = y_pred_useful - torch.mean(y_pred_useful)
	pcc = 1/((size-1)*torch.std(vy) *torch.std(vyp))*(torch.sum(vy*vyp))
	pcc = pcc.item()
	return pcc


def calculate_less_0105(y_pred_useful, y):
	'''
	inputs as tensors
	'''
	size = y.size(0)
	total05 = 0
	total1 = 0
	for a,b in zip(y, y_pred_useful):
		diff = abs(a-b)
		if diff < 1:
			total1 += 1
			if diff < 0.5:
				total05 += 1

	less1 = total1/size
	less05 = total05/size
	return (less05, less1)



def calculate_less1(y_pred_useful, y):
	less05, less1 = calculate_less_0105(y_pred_useful, y)
	return less1

def calculate_less05(y_pred_useful, y):
        less05, less1 = calculate_less_0105(y_pred_useful, y)
        return less05
