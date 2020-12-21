import numpy as np 
from multiprocessing import Pool
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

from utils import deserialize_data, resample_data, bootstrap_data
from decision_tree import Decision_tree

from tqdm import tqdm 
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold

class Random_Forest():
    '''
    Constructor for Random Forest, generate a random forest model
    Args:
        Required:
            - num_trees : number of trees we plan to use
            - total_num_feature : total number of feature as input
        Optional:
            - criterion: name for criterion (e.g. gini, entropy)
            - max_depth: maximum depth of the tree, default -1 means no limit on depth
            - max_features: maximum feature used for the tree (randomly selected), default -1 means uses all features avaliable
    '''
    def __init__(self, num_trees, total_num_feature, criterion = 'gini', max_depth = -1, max_features = -1):
        self.tree_list    = []
        self.num_trees    = num_trees
        self.criterion    = criterion
        self.max_depth    = max_depth
        self.max_features = max_features
        self.total_num_feature = total_num_feature

        # To avoid generate new trees during each training (fit) process, we make sure the forest is initialized during the __init__ method
        for i in range(self.num_trees):
            self.tree_list.append(Decision_tree(self.total_num_feature, max_features=self.max_features, max_depth=self.max_depth))

    def fit(self, data_train, label_train, field_list):
        '''
        Make the model fit the training data
        '''
        assert data_train.shape[0] == label_train.shape[0], "Number of instance for data and label not match, please check"
        assert data_train.shape[1] == len(field_list), "Number of field names doesm't match with data dimension"
        
        for i in range(self.num_trees):
            self.tree_list[i] = self.train_single_tree(i, data_train, label_train, field_list)
        
    def train_single_tree(self, idx, data_train, label_train, field_list):
        '''
        Train a single decision tree
        '''
        tree = self.tree_list[idx]
        tree.fit(data_train, label_train, field_list)

        return tree

    def fit_parallel(self, data_train, label_train, field_list, num_worker = 5):
        '''
        Make the model fit the training data, each tree is trained in parallel
        '''
        assert data_train.shape[0] == label_train.shape[0], "Number of instance for data and label not match, please check"
        assert data_train.shape[1] == len(field_list), "Number of field names doesm't match with data dimension"

        n = self.num_trees
        with ProcessPoolExecutor(max_workers=num_worker) as executor:
            idx_list = [i for i in range(self.num_trees)]
            self.tree_list = list(executor.map(self.train_single_tree, idx_list, repeat(data_train, n), repeat(label_train, n), repeat(field_list, n)))
        
    def predict(self, data, field_list):
        '''
        Make the model predict the ouput with confidence score
        '''
        assert data.shape[1] == len(field_list), "Number of field names doesm't match with data dimension"
        assert self.num_trees == len(self.tree_list), "Please first run fit before predicting"
        num_data_ins = data.shape[0]

        confidence_final = np.zeros(num_data_ins)
        prediction_final = np.zeros(num_data_ins)

        for tree in self.tree_list:
            prediction, confidence = tree.predict(data, field_list)
            confidence_final += confidence

        confidence_final /= self.num_trees
        prediction_final = np.round(confidence_final).astype(int)

        #prediction_final, confidence_final = prediction, confidence
        return prediction_final, confidence_final

    def train_tree(self, data_train, label_train, field_list):
        # data_train_bs, label_train_bs, _ = bootstrap_data(data_train, label_train)
        # self.fit_parallel(data_train_bs, label_train_bs, field_list)
        raise NotImplementedError
    
    def clear(self):
        '''
        Reset the random forest same as the status before training
        '''
        for tree in self.tree_list:
            tree.root_node = None
        
def random_forest_example(num_trees, max_features = -1, max_depth = -1):

    print("Loading data .....")
    data = deserialize_data("output/train_partial.pkl")
    print("Finished loading")
    field_list = data["field_list"]
    entry_list = data["entry_list"]

    # each row in entry_array is a fuature
    entry_array = np.array(entry_list).astype(float)
    
    ############################   Preprocess Data input #########################
    # 13: isDefault : 1.0
    # 31: policyCode : 1.0
    # 32: n0 : 0.0                Feature is not used after 32...
    
    max_feature = 32
    index_array = np.arange(max_feature)
    
    # Generate feature index array 
    target_index = 13 
    user_id_index = 0
    index_array = np.delete(index_array, [user_id_index, target_index])
    
    # Preprocess the data to remove unused input feature
    data_input = entry_array[index_array, :].T
    label = entry_array[target_index, :]
    field_input = [field_list[i] for i in index_array]


    kf = KFold(n_splits=10)
    pbar = tqdm(total=10)
    sore_sum = 0
    acc_sum = 0 
    random_forest = Random_Forest(num_trees, data_input.shape[1], max_features=max_features, max_depth=max_depth)

    for train_index, test_index in kf.split(data_input):

        data_train, label_train = data_input[train_index], label[train_index]

        random_forest.fit_parallel(data_train, label_train, field_input)

        data_eval, label_eval = data_input[test_index], label[test_index]
        prediction, confidence = random_forest.predict(data_eval, field_input)
        accuracy = np.where(label_eval == prediction)[0].shape[0]/label_eval.shape[0]
        acc_sum += accuracy
        
        # Calculate final scores
        fpr, tpr, thresholds = roc_curve(label_eval, confidence)
        score = auc(fpr, tpr)
        sore_sum += score
        pbar.update(1)

        random_forest.clear()

    # print(sore_sum/10)
    # print(acc_sum/10)
    pbar.close()

    return sore_sum/10, acc_sum/10

if __name__ == "__main__":

    num_exp = 30
    score_exp = np.zeros(num_exp)
    acc_exp = np.zeros(num_exp)
    for i in range(num_exp):
        score_i, acc_i = random_forest_example(10, max_depth = 7, max_features = 5)
        print(score_i, acc_i)
        score_exp[i] = score_i
        acc_exp[i] = acc_i
    
    result = [score_exp, acc_exp]
    import pickle
    with open("output/rf_result.pkl", "wb") as f:
        pickle.dump(result, f)



    # import pickle
    # with open("output/rf_result.pkl", "rb") as f:
    #     result = pickle.load(f)
    # print(np.mean(result[0]))
    # print(np.mean(result[1]))
    # print(np.std(result[0]))
    # print(np.std(result[1]))
    # print("ok")

    # random_forest_example(10, max_depth = 7, max_features = 5)