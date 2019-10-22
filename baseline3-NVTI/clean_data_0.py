data_pair = np.loadtxt('data/hotel_data_pair_105K.txt', delimiter=',', dtype = np.str).tolist()
n = len(data_pair)
n_train_ratio =  0.8
n_val_ratio = 0.1
n_test_ratio =  0.1

A = data_pair[0:int(0.8*len(data_pair))]
B = data_pair[int(n_train_ratio*n): int((n_train_ratio+n_val_ratio)*n)]
C = data_pair[int((n_train_ratio+n_val_ratio)*n):n]


with open('data/hotel_data_pair_105K_train.txt', 'w') as f:  
	for item in A:
		f.writelines(item[0])
		f.writelines(',')
		f.writelines(item[1])
		f.writelines('\n')


with open('data/hotel_data_pair_105K_val.txt', 'w') as f:  
	for item in B:
		f.writelines(item[0])
		f.writelines(',')
		f.writelines(item[1])
		f.writelines('\n')


with open('data/hotel_data_pair_105K_test.txt', 'w') as f:  
	for item in C:
		f.writelines(item[0])
		f.writelines(',')
		f.writelines(item[1])
		f.writelines('\n')




