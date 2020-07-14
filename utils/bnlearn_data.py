import numpy as np
import pandas as pd
import tensorflow as tf

import urllib.request
import zipfile
import csv
import os

def to_onehot(inp):
  s = pd.Series(inp)
  out = pd.get_dummies(s)
  return out

def split_list(a_list):
    size = len(a_list)//4
    return a_list[:len(a_list)-size], a_list[len(a_list)-size:]


def load_dnn_prob(train_prob, test_prob, features):

  XA_train_in, XA_test_in = split_list(train_prob)
  XA_train_out, XA_test_out = split_list(test_prob)
#  XA_train_out2, XA_test_out2 = split_list(list(predict_test2))

  # print(len(XA_train_in))
  # print(len(XA_train_out))
  # Splitting to make the number of member samples equal to non-members
  XA_train_in = XA_train_in[:len(XA_train_out)]
  XA_test_in = XA_test_in[:len(XA_test_out)]

  # Creating the labels for the training points of the attacker model (1--> member and 0--> non-member)
  y_att_train = [1,0]
  y_att_test = [1,0]

  y_att_train = np.pad(y_att_train, (len(XA_train_in)-1,len(XA_train_out)-1),'edge')
  y_att_train = to_onehot(y_att_train) # One-hot encoding for the labels (members/non-members)

  y_att_test = np.pad(y_att_test, (len(XA_test_in)-1, len(XA_test_out)-1), 'edge')
  y_att_test = to_onehot(y_att_test)

  # Combining members and non-members into training and test points
  XA_train_in.extend(XA_train_out)
  XA_test_in.extend(XA_test_out)
  # rint(y_att_test[0][5])
  import csv
  with open("datasets/child/Age/dnn_output.csv", "w") as f:
      writer = csv.writer(f)
      for i in range(len(XA_train_in)):
          dict_val = XA_train_in[i]
          arr1 = dict_val["probabilities"]
          arr2 = dict_val["logits"]
          arr3 = dict_val["class_ids"]
          arr = np.concatenate((arr1,arr2,arr3,[y_att_train[0][i]]))
          #arr = arr1+arr2+arr3 + [y_att_train[0][i]]
          writer.writerow(arr)
      for i in range(len(XA_test_in)):
          dict_val = XA_test_in[i]
          arr1 = dict_val["probabilities"]
          arr2 = dict_val["logits"]
          arr3 = dict_val["class_ids"]
          arr = np.concatenate((arr1,arr2,arr3,[y_att_test[0][i]]))
          # print(arr)
          writer.writerow(arr)
  # print(XA_train_in[:10])
  # print(XA_test_in[:10])
  # Creating the inputs in the desired format for the attacker estimator
  my_feature_columns = []
  X_att_train = {}
  X_att_test = {}

  for key in features:
    my_feature_columns.append(tf.feature_column.numeric_column(key=str(key)))
    X_att_train[str(key)] = []
    X_att_test[str(key)] = []


  key_length = len(list(X_att_train.keys()))

  for key,k in zip(X_att_train.keys(),range(key_length)):
    for i in range(len(XA_train_in)):
      val = list(XA_train_in[i]['probabilities'])
      X_att_train[str(key)].append(val[k])

  for key,k in zip(X_att_train.keys(),range(key_length)):
    for i in range(len(XA_test_in)):
      val = list(XA_test_in[i]['probabilities'])
      X_att_test[str(key)].append(val[k])

  X_att_train = pd.DataFrame(X_att_train)
  X_att_test = pd.DataFrame(X_att_test)

  # Randomize the indices
  ran_idx = np.random.permutation(X_att_train.index)
  X_att_train = X_att_train.reindex(ran_idx)
  y_att_train = y_att_train.reindex(ran_idx)

  ran_idx = np.random.permutation(X_att_test.index)
  X_att_test = X_att_test.reindex(ran_idx)
  y_att_test = y_att_test.reindex(ran_idx)

  return (X_att_train, y_att_train), (X_att_test, y_att_test), my_feature_columns

def load_bnet_prob(dataset_name, num_examples, output_name, dist):
  data_dir = "datasets/"+dataset_name+"/"+ output_name+"/"+ str(num_examples)
  all_in_data = pd.read_csv(data_dir+"/"+dataset_name+"_train_prob.csv", sep=",")#, header=None)

  if dist == 1:
    all_out_data = pd.read_csv(data_dir+"/"+dataset_name+"_test_prob.csv", sep=",")
  else:
    all_out_data = pd.read_csv(data_dir+"/"+dataset_name+"_test_prob_"+str(dist)+".csv", sep=",")

  Xtrain_out = all_out_data.sample(frac=0.75)
  Xtest_out = all_out_data.drop(Xtrain_out.index)

  Xtrain_in = all_in_data.sample(n=len(Xtrain_out))
  Xtest_in = all_in_data.drop(Xtrain_in.index).sample(n=len(Xtest_out))

  train_y = [1,0]
  test_y = [1,0]

  train_y = np.pad(train_y, (len(Xtrain_in)-1,len(Xtrain_out)-1),'edge')
  s = pd.Series(train_y)
  train_y = pd.get_dummies(s) # One-hot encoding for the labels (members/non-members)

  test_y = np.pad(test_y, (len(Xtest_in)-1, len(Xtest_out)-1), 'edge')
  s = pd.Series(test_y)
  test_y = pd.get_dummies(s)

  train_x = Xtrain_in.append(Xtrain_out)
  test_x = Xtest_in.append(Xtest_out)

  # Randomize the indices
  ran_idx = idx = np.random.permutation(train_x.index)
  train_x.index = ran_idx
  train_y.index = ran_idx

  ran_idx = idx = np.random.permutation(test_x.index)
  test_x.index = ran_idx
  test_y.index = ran_idx

  with open(data_dir+"/"+dataset_name+"_bnet_acc.txt",) as f:
    first_line = f.readline()

  accuracy = first_line.split(',')
  for i in accuracy:
    print(i)

  return (train_x, train_y), (test_x, test_y)

def load_data(dataset_name, num_examples, output_name, noise):
  data_dir = "datasets/"+dataset_name+"/"+ output_name+"/"+ str(num_examples)
  train = pd.read_csv(data_dir+ "/" +dataset_name+"_train_data.csv", sep=",")#, header=None)
  test = pd.read_csv(data_dir+ "/" +dataset_name+"_test_data.csv", sep=",")

  test_dist2_x = []
  test_dist2_y = []

  for i in noise:
    test2 = pd.read_csv(data_dir+ "/" +dataset_name+"_test_data_"+str(i)+".csv", sep=",")  
    test2_x, test2_y = test2, test2.pop(output_name)
    test2_y = to_onehot(test2_y)
    test_dist2_x.append(test2_x)
    test_dist2_y.append(test2_y)

  train_x, train_y = train, train.pop(output_name)
  train_y = to_onehot(train_y) # Create one-hot encoding for the labels

  test_x, test_y = test, test.pop(output_name)
  test_y = to_onehot(test_y)


  # with open(data_dir+"/"+dataset_name+"_train_prob.csv",) as f:
  #   first_line = f.readline()

  # N_classes = len(first_line.split(','))

  return (train_x, train_y), (test_x, test_y), (test_dist2_x, test_dist2_y), len(test_y.keys())

def train_input_fn(features, labels, batch_size):
  """An input function for training"""
  # Convert the inputs to a Dataset.


  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  # Shuffle, repeat, and batch the examples.
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)

  # Return the dataset.
  return dataset


def eval_input_fn(features, labels, batch_size):
  """An input function for evaluation or prediction"""
  features=dict(features)
  if labels is None:
    # No labels, use only features.
    inputs = features
  else:
    inputs = (features, labels)

  # Convert the inputs to a Dataset.
  dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Shuffle, repeat, and batch the examples.
  #dataset = dataset.shuffle(10).repeat().batch(batch_size)

  # Batch the examples
  assert batch_size is not None, "batch_size must not be None"
  dataset = dataset.batch(batch_size)

  # Return the dataset.
  return dataset

if __name__=="__main__":
  load_data("sachs", 1000, "Mek")

