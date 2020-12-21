import pickle

import numpy as np 
from tqdm import tqdm 

def load_csv_data(file_name, progress_bar = True, max_ins = -1):
    '''
    Load from csv file and then store them in row-wise and column-wise to reduce access time
    Args:
        file_name: data path for csv file
        progress_bar: False to disable the load progress bar
        max_ins: maximum data number to load (load small portion for debugging)
    Return:
        field_list: list for field name and their position
        line_list: store by row wise
        entry_list: store by column wise
    '''
    field_list = []
    entry_list = []
    line_list = []
    if progress_bar: 
        print("Loading data from csv file")
        pbar = tqdm(total=int(8e5))
    
    with open(file_name, "r") as f:
        line = f.readline()
        
        field_list = line.rstrip().split(",")
        num_field = len(field_list)
        #print(num_field)
        #field_dict = {i:field_list[i] for i in range(num_field)}
        entry_list = [[] for i in range(num_field)]
         
        line = f.readline()
        k = 0
        while line and (max_ins < 0 or k < max_ins):
            line_item = line.rstrip().split(",")
            line_list.append(line_item)
            #print(line_item)
            for i in range(num_field):
                entry_list[i].append(line_item[i])

            if progress_bar: pbar.update(1)
            line = f.readline()
            k += 1
            
    return field_list, line_list, entry_list

def serialize_result(file_path, **kwargs):
    '''
    Save data to pickle file
    '''
    with open(file_path, "wb") as f:
        pickle.dump(kwargs, f)
    
    return True

def deserialize_data(file_path):
    '''
    Load data from pickle data
    '''
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    return data

def visualize_histogram(value_list, feature_name="", fig_save_path = "temp.png", num_bin = 30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt 
    '''
    Visualize the histogram of values and extract them 
    '''
    fig, ax = plt.subplots(1)
    value_dict = {}
    
    value_array = np.array(value_list).astype(float)
    ax.hist(value_array, bins=30)
    ax.set_title(feature_name)

    fig.show()
    fig.savefig(fig_save_path)
    plt.close(fig=fig)

def preprocess_string(**kwargs):
    '''
    Currently, we support 
        5: grade : E
        6: subGrade : E2
        8: employmentLength : 2 years
        12: issueDate : 2014-07-01
        29: earliesCreditLine : Aug-2001
    '''
    res_dict = {}
    print("Preprocessing string...")
    for key, value in tqdm(kwargs.items()):
        if key == "grade" or key == "subGrade":
            key_set = set()
            for i in value:
                key_set.add(i)
            key_list_sorted = list(sorted(key_set))
            key_hash_value = { key_list_sorted[i]:i for i in range(len(key_list_sorted)) }
            
            ranked_value = []
            for v in value:
                ranked_value.append(key_hash_value[v])
    
        elif key == "employmentLength":        
            key_list_sorted = ['', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
            key_hash_value = { key_list_sorted[i]:i-1 for i in range(len(key_list_sorted)) }
            ranked_value = []
            for v in value:
                ranked_value.append(key_hash_value[v])
        
        elif key == "issueDate":
            ranked_value = []
            for v in value:
                v_split = v.split("-")
                v_value = int(v_split[0]) *10000 + int(v_split[1])*100 + int(v_split[2])
                ranked_value.append(v_value)

        elif key == "earliesCreditLine":
            month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May','Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            key_hash_value = { month_list[i]:i+1 for i in range(len(month_list)) }

            ranked_value = []
            for v in value:
                v_split = v.split("-")
                v_value = key_hash_value[v_split[0]] + int(v_split[1])*100
                ranked_value.append(v_value)
        else:
            raise NotImplementedError

        # Solve clean data to res_dict
        res_dict[key] = ranked_value

    return res_dict

def filter_out_empty_element(entry_list):
    '''
    Args:
        entry_list
    Return:
        result_array: array replace any empty space with -1
    '''
    result_list = []
    for i,v in enumerate(entry_list):
        try:
            result_list.append(float(v))
        except ValueError:
            if v == '':
                result_list.append(-1)
            else:
                raise RuntimeError
    
    return result_list

def resample_data(data_input, label, num_each_category = int(5e3), replacement = True):
    '''
    Resample the data, to make sure the class is balanced
    Args:
        data_input: (n, num_features)
        label : (n,)
        num_each_category: number of data instances for each category
        replacement: if true, the data is resampled with replacement
    Return:
        data_resample : (2*num_each_category, num_features)
        label_resample : (2*num_each_category,)
        selected_index : selected indexes from the original array
    '''
    #num_train_ins  = num_each_category
    assert data_input.shape[0] == label.shape[0]
    selected_index = np.zeros(2*num_each_category).astype(int)

    non_zero_index = np.where(label == 1)[0]
    rand_choice    = np.random.choice(non_zero_index.shape[0], num_each_category, replace=replacement)
    selected_index[:num_each_category] = non_zero_index[rand_choice]

    zero_index     = np.where(label == 0)[0]
    # For negative samples, replacement is not necessary 
    rand_choice    = np.random.choice(zero_index.shape[0], num_each_category, replace=False)
    selected_index[num_each_category:] = zero_index[rand_choice]

    # randm suffle the selected index and use it to resample the dataset
    np.random.shuffle(selected_index)
    data_resample  = data_input[selected_index, :]
    label_resample = label[selected_index]
    
    return data_resample, label_resample, selected_index


def bootstrap_data(data_input, label):
    '''
    Bootstrap the data
    Args:
        data_input, label
    Return:
        data_input_res, label_res
        oob_index_array: out-of-bag index array, useful for calculating OOB error
    '''
    assert data_input.shape[0] == label.shape[0]
    num_ins = data_input.shape[0]
    rand_choice  = np.random.choice(num_ins, num_ins, replace=True)
    
    data_input_res = data_input[rand_choice]
    label_res = label[rand_choice]

    selected_set = set(rand_choice)
    entire_set = {i for i in range(num_ins)}
    oob_set = entire_set - selected_set
    oob_index_array = np.array(list(oob_set))

    return data_input_res, label_res, oob_index_array
#################              code for unit test           ############################

def save_data_test():
    file_name = "data/train.csv"
    field_list, line_list, entry_list = load_csv_data(file_name)#, max_ins = int(5e4))
    
    '''
    5: grade : E
    6: subGrade : E2
    8: employmentLength : 2 years
    12: issueDate : 2014-07-01
    29: earliesCreditLine : Aug-2001
    '''
    field_contain_string_list = [5,6,8,12,29]
    value_pair = {field_list[k]:entry_list[k] for k in field_contain_string_list}
    res_dict = preprocess_string(**value_pair)

    # Update existing array
    for field_index in field_contain_string_list:
        field_name = field_list[field_index]
        entry_list[field_index] = res_dict[field_name]

    field_used = [i for i in range(len(field_list))]
    entry_array_list = [] 
    for i in field_used:
        entry_list_i = entry_list[i]
        entry_list[i] = filter_out_empty_element(entry_list_i)
    
    print("Saving data ....")
    serialize_result("output/train.pkl", field_list = field_list, line_list = line_list, entry_list = entry_list)

def save_data_partial_test():
    print("Loading data .....")
    data = deserialize_data("output/train.pkl")
    print("Finished loading")

    line_list = data["line_list"]
    field_list = data["field_list"]
    entry_list = data["entry_list"]

    num_ins = len(line_list)
    num_sample = int(5e4)
    index_selected = np.random.choice(num_ins, num_sample, replace=False).astype(int).tolist()

    line_list = [line_list[k] for k in index_selected]
    num_field = len(entry_list)
    for i in range(num_field):
        entry_list[i] = [entry_list[i][k] for k in index_selected]

    print("Saving partial data ....")
    serialize_result("output/train_partial.pkl", field_list = field_list, line_list = line_list, entry_list = entry_list)

def vis_test():
    print("Loading data .....")
    data = deserialize_data("output/train.pkl")
    print("Finished loading")

    line_list = data["line_list"]
    field_list = data["field_list"]
    entry_list = data["entry_list"]

    print("visulizing test")
    for i, f in enumerate(field_list):
        visualize_histogram(entry_list[i], feature_name = f, num_bin=30, fig_save_path="output/" + f + ".png")
    
if __name__ == "__main__":
    #save_data_test()
    #load_test()
    #save_data_partial_test()
    pass
    

    




    '''trash'''
    # line_list = data["line_list"]
    # field_list = data["field_list"]
    # entry_list = data["entry_list"]
    
    
        # print("Saving data ....")
    # serialize_result("output/train.pkl", field_list = field_list, line_list = line_list, entry_list = entry_list)
    # print("Loading data .....")
    # data = deserialize_data("output/train.pkl")
    # print("Finished loading")
    # line_list = data["line_list"]
    # field_list = data["field_list"]
    # entry_list = data["entry_list"]
    #visualize_histogram(entry_list[13], feature_name = field_list[13], num_bin=2)
    
    #clean_data(grade = entry_list[5])

    # num_field = len(field_list)
    # print(num_field)
    # field_used = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    # c_list = []
    # for i in field_used:
    #     try:
    #         float(entry_list[i][0])
    #         print("{}: {} : {}".format(i, field_list[i], float(entry_list[i][0])))
    #     except ValueError:
    #         c_list.append(i)
    # print("-"*10)
    # for i in c_list:
    #     print("{}: {} : {}".format(i, field_list[i], entry_list[i][0]))
    
    #print(entry_list[8][0:50])
    #res_dict = preprocess_string(earliesCreditLine = entry_list[29])

