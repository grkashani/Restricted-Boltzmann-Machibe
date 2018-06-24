from __future__ import print_function
import numpy as np
import sys
import pickle

class RBM:

  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    np_rng = np.random.RandomState()
    self.weights = np.asarray(np_rng.uniform(low=-0.09,high=0.09,size=(self.num_visible,self.num_hidden)))
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def e(self,s='',f=0):
    print('\n',s)
    if f==0:
      sys.exit(0)

  def activation_function(self, x):
    return 1.0 / (1 + np.exp(-x))

  def test(self,user):
    user_h = self.run_visible(user)
    user_v = self.run_hidden(user_h)
    print("-----------------------------")
    print('Hidn',' = ',user_h) 
    print('Real',' = ',user)
    print('Calc',' = ',user_v)

  def run_visible(self, data):
    rows = data.shape[0]
    hidden_states = np.ones((rows, self.num_hidden + 1))
    data = np.insert(data, 0, 1, axis = 1)
    hidden_activations = np.dot(data, self.weights)
    hidden_probs = self.activation_function(hidden_activations)
    hidden_states[:,:] = hidden_probs > np.random.rand(rows, self.num_hidden + 1)
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  def run_hidden(self, data):
    rows = data.shape[0]
    visible_states = np.ones((rows, self.num_visible + 1))
    data = np.insert(data, 0, 1, axis = 1)
    visible_activations = np.dot(data, self.weights.T)
    visible_probs = self.activation_function(visible_activations)
    visible_states[:,:] = visible_probs > np.random.rand(rows, self.num_visible + 1)
    visible_states = visible_states[:,1:]
    return visible_states
    
  def get_input_from_output(self,out_1,rows):
      hidden_states = out_1 > np.random.rand(rows, self.num_hidden + 1)
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self.activation_function(visible_activations)
      visible_probs[:,0] = 1 
      return visible_probs
  def cal_weights(self,data):    
      hidden_activations = np.dot(data,self.weights)
      hidden_probs = self.activation_function(hidden_activations)
      associations = np.dot(data.T,hidden_probs)
      return hidden_probs,associations
  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
    rows = data.shape[0]
    data = np.insert(data, 0, 1, axis = 1)
    for epoch in range(max_epochs):
      out_1,w1 = self.cal_weights(data)
      data_from_output = self.get_input_from_output(out_1,rows)
      out_2,w2 = self.cal_weights(data_from_output)
      
      self.weights += learning_rate * (w1 - w2)
    
    error = np.sum((data - data_from_output) ** 2)
    return epoch,error

  def creat_random_data(self,digits):
    test = []
    largest_number = 2 ** digits 
    for i in range(largest_number):
      test.append((np.array([int(x) for x in bin(i)[2:].zfill(digits)])))
    return(np.array(test))



if __name__ == '__main__':
  
  n_input = 8
  n_output = 9
  
  my_rbm = RBM(num_visible = n_input, num_hidden =n_output)
  training_data = my_rbm.creat_random_data(n_input) 
  print('nunber_data = ',len(training_data),'/',2 ** n_input)
  epoch,error = my_rbm.train(training_data, max_epochs = 999999,learning_rate=0.001)
  print('after',epoch,'times, error is = ',error)
  with open("model.rbm", "wb") as f:
    pickle.dump(my_rbm, f,pickle.HIGHEST_PROTOCOL)

  my_rbm.test(np.array([[float(x) for x in bin(2)[2:].zfill(n_input)]]))

