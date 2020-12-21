import numpy as np 
from utils import deserialize_data, resample_data
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold
from tqdm import tqdm 

# Use the same random seed for debugging
# np.random.seed(1)

class Decision_node():
    def __init__(self, feature_index = -1, feature_name = None, value = -1, labels = [], l_b = None, r_b = None, conf = -1):
        self.value = value                    # split value
        self.feature_index = feature_index    # best feature index in current data subset
        self.feature_name = feature_name      # corresponded feature name
        self.labels = labels                  # labels for current split
        self.l_b = l_b                        # left branch of decision tree, where <= the value
        self.r_b = r_b                        # right branch of decisoin tree, where > the value
        self.conf = conf                      # Calculate the percentage of non zero label, use the as the confidence of prediction

class Decision_tree():
    '''
    Construct a decision tree object
    '''
    def __init__(self, total_num_feature, criterion = 'gini', max_depth = -1, max_features = -1, num_percentile = 10):
        '''
        Args:
        Required:
            - total_num_feature: total number of features we plan to use as input
        Optional:
            - criterion: name for criterion (e.g. gini, entropy)
            - max_depth: maximum depth of the tree, default -1 means no limit on depth
            - max_features: maximum feature used for the tree (randomly selected), default -1 means uses all features avaliable
            - num_percentile: if unique value of data exceed the 2*percentile, then use this to split values, or use percentage
        '''
        self.root_node = None
        self.selected_feature = None
        self.total_num_feature = total_num_feature

        # parameters
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.num_percentile = num_percentile

        # Only use a subset of the features if max_features is specified
        if self.max_features > 0  and  total_num_feature > self.max_features:
            self.selected_feature = np.random.choice(total_num_feature, self.max_features, replace=False)
        else:
            self.selected_feature = np.arange(total_num_feature)
     
    def fit(self, data, label, field_list):
        '''
            Fit decision tree to training data
            Args:
                data: input data, size is (num_instance, num_feature)
                label: labels for these input data
                filed_list: meaning for each column
        '''

        num_instance, num_feature = data.shape
        assert num_instance == label.shape[0], "Please make sure the number of labels is the same as number of data point instances"
        assert num_instance > 0, "We need at least 1 data point to do this"
        assert num_feature == self.total_num_feature, "Make sure the feature number is the same as you planned to use"

        # Only use a subset of the features if max_features is specified
        if self.max_features > 0  and  num_feature > self.max_features:
            data = data[:, self.selected_feature]
            field_list = [field_list[i] for i in self.selected_feature]
            
        # print(self._cal_current_stats(self.criterion, label))
        self.root_node = self._grow_tree(data, label, self.max_depth)

        Decision_tree.translate_feature(self.root_node, field_list)
        # Decision_tree.print_level_order(self.root_node)
        
    def predict(self, data, field):
        '''
        predict data instance given the values and their fields
        Args:
            data: input data, size is (num_instance, num_feature)
            filed_list: meaning for each column
        Return:
            prediction:
            confidence: confidence score, if <0.5, negative, if >0.5 positive 
        '''
        assert self.root_node is not None,  "Please first use predict method to generate a tree"
        assert self.root_node.feature_name is not None, "Please Translate the feature index to feature name"
        
        feature2index_dict = {v:i for i, v in enumerate(field)}
        num_ins = data.shape[0]

        prediction = -np.ones(num_ins)
        confidence = np.zeros(num_ins)

        for i in range(num_ins):
            current_node = self.root_node
            data_i = data[i]
            while(current_node.feature_index != -1):
                current_feature_index = feature2index_dict[current_node.feature_name]
                if data_i[current_feature_index] <= current_node.value:
                    current_node = current_node.l_b
                else:
                    current_node = current_node.r_b
            if current_node.conf <=0.5:
                prediction[i] = 0
                confidence[i] = current_node.conf
            else:
                prediction[i] = 1
                confidence[i] = current_node.conf


        return prediction, confidence
    
    def _grow_tree(self, data, label, depth):
        '''
        Grow current tree based on the input node
        '''
        # Depth constraint us from dividing further
        if depth == 0:
            #value_set, unique_indices = np.unique(label, return_inverse=True)
            return Decision_node(labels = label, conf = np.count_nonzero(label)/len(label))
        # No data point remaining, 
        if data.shape[0] == 0:
            raise NotImplementedError
        
        current_state = self._cal_current_stats(self.criterion, label)

        best_gain = 0
        best_split_param = (0,0) # (feature index,  split value)
        best_split = None  # (left index, right index)
        num_feature = data.shape[1]

        for i in range(num_feature):

            best_impurity, best_split_value, best_set_i = self._find_best_split(data[:, i], label)

            if len(best_set_i) == 1:
                continue
            
            
            gain = current_state - best_impurity
            
            if gain > best_gain:
                best_gain = gain
                best_split_param = (i, best_split_value)
                best_split = best_set_i
                
        if best_gain > 0:
            # construct left branch and right branch
            left_branch = self._grow_tree(data[best_split[0]], label[best_split[0]], depth - 1)
            right_branch = self._grow_tree(data[best_split[1]], label[best_split[1]], depth - 1)
            
            # connect them together
            current_node = Decision_node(feature_index= best_split_param[0],
                                value=best_split_param[1], 
                                labels = label,
                                l_b = left_branch,
                                r_b = right_branch,
                                conf = np.count_nonzero(label)/len(label)
                                )
            return current_node
        else:
            #value_set, unique_indices = np.unique(label, return_inverse=True)
            return Decision_node(labels = label, conf = np.count_nonzero(label)/len(label))
     
    def _find_best_split(self, input_feature, label):
        '''
        Args:
            input_feature: current feature input
            label: current label input
        Return:
            best_impurity: impurity value based on the default criterion, less is better
            best_value   : best value to split the feature
            best_set     : the splitted set based on your value
        '''
        #value_set, unique_indices = np.unique(data[:, i], return_inverse=True)
        best_value = 0
        best_impurity = 1
        best_set = None
        
        value_list = np.unique(input_feature, return_inverse=False)
        num_value = len(value_list)
        
        # if value set is too large, we may need some speed up by split the values by percentile
        if num_value > 2 * self.num_percentile :
            # overwrite num_value and value_list 
            num_value = self.num_percentile
            value_list = []
            for i in range(num_value + 1):
                value_list.append(np.percentile(input_feature, (i/num_value)*100))
    
        if num_value > 1:
            for i in range(num_value - 1):
                l_set = np.where(input_feature <= value_list[i])
                r_set = np.where(input_feature > value_list[i])

                l_state = self._cal_current_stats(self.criterion, label[l_set])
                r_state = self._cal_current_stats(self.criterion, label[r_set])

                num_l_set = len(l_set[0])
                num_r_set = len(r_set[0])
                average_state = (l_state*num_l_set + r_state*num_r_set)/(num_l_set + num_r_set)
                
                if best_impurity > average_state:
                    best_impurity = average_state
                    best_value = value_list[i]
                    best_set = (l_set, r_set)
        else:
            return 1, value_list[0], ([i for i in range(len(input_feature))], )

        return best_impurity, best_value, best_set
    
    def _cal_current_stats(self, state_function, labels):
        '''
        Calculate current entropy or impurity given then state function and labels
        Args:
            state_function: e.g. impurity function or entropy function
            labels: target lable of each data instance
            split_value: 
        '''
        if state_function == "gini":
            return Decision_tree.gini_function(labels)
        else:
            raise NotImplementedError

    @staticmethod
    def gini_function(label_array):
        '''
        Calculate gini impurity of current target set
        Args:
            label_array
        Return:
            gini impurity
        '''
        value_set = np.unique(label_array)
        num_label = label_array.shape[0]

        return 1.0 - sum([(len(label_array[label_array == c]) / num_label) ** 2.0 for c in value_set]) 

    @staticmethod
    def print_level_order(tree_root):
        '''
        Traverse the whole tree with BFS method
        Args:
            tree_root: root node of the tree
        '''
        queue_p = []
        queue_p.append(tree_root)
        k = 1

        print("-"*10 , "  START PRINT TREE  ", "-"*10)
        while queue_p:
            feature_id_list = []
            num_split_list = []
            print("-"*5, "  level   ", k, "-"*5)
            for i in range(len(queue_p)):
                s = queue_p.pop(0)
                if s.l_b is not None:
                    queue_p.append(s.l_b)
                if s.r_b is not None:
                    queue_p.append(s.r_b)
                feature_id_list.append(s.feature_name)
                num_split_list.append(s.conf)
            print(feature_id_list)
            print(num_split_list)
            
            k += 1
        print("-"*15 , "  END  ", "-"*15)

    @staticmethod
    def translate_feature(tree_root, field_list):
        '''
        Tranaslate feature index to feature name in each node of the tree 
        Args:
            tree_root: root node of the tree
            field_list: field name list
        '''
        queue_p = []
        queue_p.append(tree_root)
        # total_ins = 0
        # total_imp = 0

        while queue_p:
            s = queue_p.pop(0)
            if s.l_b is not None:
                queue_p.append(s.l_b)
            if s.r_b is not None:
                queue_p.append(s.r_b)
            
            if s.feature_index >= 0:
                s.feature_name = field_list[s.feature_index]
            else:
                s.feature_name = "NA"

                # r1 = np.count_nonzero(s.labels)/len(s.labels)
                # imp = 1 - r1**2 - (1-r1)**2
                # total_ins += len(s.labels)
                # total_imp += imp*len(s.labels)
        
        # print(total_ins)
        # print(total_imp/total_ins)
        # print("x"*20)
    


########## unit test  #########
def test1():
    # Gini function
    x = np.random.choice(2,10)
    print(x)
    print(Decision_tree.gini_function(x))

#########  Example usage for Decision Tree ##########
def generate_decision_tree(max_features = -1, max_depth = 20, selected_feature = None):
    
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

    # Only select a subset of features for training
    if selected_feature is not None:
        data_input = data_input[:, selected_feature]
        field_input = [field_input[i] for i in selected_feature]

    kf = KFold(n_splits=10)
    pbar = tqdm(total=10)
    sore_sum = 0
    acc_sum = 0

    tree = Decision_tree(data_input.shape[1], max_features=max_features, max_depth=max_depth)
    #############################     Evaluate decision tree with 10-Fold      ##############################
    for train_index, test_index in kf.split(data_input):
        data_train, label_train = data_input[train_index], label[train_index]
        # Resamle data to make sure the class is balanced
        # data_train, label_train, _ = resample_data(data_train, label_train, num_each_category = int(3e4), replacement = True) # Upsample
        # data_train, label_train, _ = resample_data(data_train, label_train, num_each_category = int(8e3), replacement = False) # Downsample
        
        tree.fit(data_train, label_train, field_input)

        data_eval, label_eval = data_input[test_index], label[test_index]
        prediction, confidence = tree.predict(data_eval, field_input) 
        accuracy = np.where(label_eval == prediction)[0].shape[0]/label_eval.shape[0]
        acc_sum += accuracy
        # print(prediction)
        # print(confidence)
        
        # Calculate final scores
        fpr, tpr, thresholds = roc_curve(label_eval, confidence)
        score = auc(fpr, tpr)
        sore_sum += score
        # print(tree.selected_feature)
        pbar.update(1)
    
    pbar.close()
    # print(sore_sum/10)
    # print(acc_sum/10)

    return sore_sum/10, acc_sum/10

if __name__ == "__main__":

    num_exp = 30
    score_exp = np.zeros(num_exp)
    acc_exp = np.zeros(num_exp)
    for i in range(num_exp):
        score_i, acc_i = generate_decision_tree(max_depth = 6, max_features = -1)
        print(score_i, acc_i)
        score_exp[i] = score_i
        acc_exp[i] = acc_i
    
    result = [score_exp, acc_exp]
    import pickle
    with open("output/dt_full_result.pkl", "wb") as f:
        pickle.dump(result, f)
    

    # generate_decision_tree(max_depth = 6, max_features = -1)
    # generate_decision_tree(max_depth = 6, max_features = 5, selected_feature = [4, 5, 12, 15, 29])
    
    # import pickle
    # with open("output/dt_result.pkl", "rb") as f:
    #     result = pickle.load(f)
    # print(np.mean(result[0]))
    # print(np.mean(result[1]))
    # print(np.std(result[0]))
    # print(np.std(result[1]))
    # print("ok")
