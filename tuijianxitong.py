# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def get_data(data_path):
    # extract_data
    head=['user_id','item','rating','timsap']
    f=pd.read_csv(data_path,sep='\t',names=head)
    # data_sum
    sum_user=f['user_id'].unique().shape[0]
    sum_items=f['item'].unique().shape[0]
    #make_matrix
    rating=np.zeros((sum_user,sum_items))
    for row in f.itertuples():
        rating[row[1]-1,row[2]-1]=row[3]
    x_train,x_test=train_test_split(rating,test_size=0.25,random_state=33)    
    return rating,x_train,x_test   
def model_train(x_train,kind='user',el=1e-9):
   #similarity
    if kind=='user':
       sim=x_train.dot(x_train.T)+el
   elif kind=='item':
       sim=x_train.T.dot(x_train)+el
   norm=np.array([np.sqrt(np.diagonal(sim))])
   return (sim / norm / norm.T)
def model_predict(rating,sim,kind='user'):
    if kind=='user':
        return sim.dot(rating)/np.array([np.abs(sim).sum(axis=1)]).T
    elif kind=='item':
        return rating.dot(sim)/np.array([np.abs(sim).sum(axis=1)])
def model_mse(pre,actual):
    #mse
    pre=pre[actual.nonzero()].flatten()
    actual=actual[actual.nonzero()].flatten()
    return mean_squared_error(pre, actual)
def model_nobias(rating,sim,kind='user'):
    if kind=='user':
        user_bias=rating.mean(axis=1)
        rating=(rating-user_bias[:,np.newaxis]).copy()
        pre=sim.dot(rating)/np.array([np.abs(sim).sum(axis=1)]).T
        pre+=user_bias[:,np.newaxis]
    elif kind=='item':
        item_bias=rating.mean(axis=0)
        rating=(rating - item_bias[np.newaxis, :]).copy()
        pre = rating.dot(sim) / np.array([np.abs(sim).sum(axis=1)])
        pre += item_bias[np.newaxis, :]
    return pre
def predict_top(rating,sim,kind='user',k=20):
    pre=np.zeros(rating.shape)
    if kind=='user':
        user_bias=rating.mean(axis=1)
        rating=(rating-user_bias[:,np.newaxis]).copy()
        for i in range(rating.shape[0]):
            #columns_index
            top_k=[np.argsort(sim[:,i])[:-k-1:-1]]
            for j in range(rating.shape[1]):
                pre[i, j] = sim[i, :][top_k].dot(rating[:, j][top_k]) 
                pre[i, j] /= np.sum(np.abs(sim[i, :][top_k]))
        pre += user_bias[:, np.newaxis]
    if kind == 'item':
        for j in range(rating.shape[1]):
            top_k_items = [np.argsort(sim[:,j])[:-k-1:-1]]
            for i in range(rating.shape[0]):
                pre[i, j] = sim[j, :][top_k_items].dot(rating[i, :][top_k_items].T) 
                pre[i, j] /= np.sum(np.abs(sim[j, :][top_k_items]))
    return pre
def model_plot(rating,x_train,x_test,user_sim,item_sim):
    ks=[5,15,20,30,40,50]
    user_train_mse=[]
    user_test_mse=[]
    item_train_mse=[]
    item_test_mse=[]
    for k in ks:
        user_pred=predict_top(x_train,user_sim,kind='user',k=k)
        item_pred=predict_top(x_train,item_sim,kind='item',k=k)   
        user_train_mse += [model_mse(user_pred, x_train)]
        user_test_mse += [model_mse(user_pred, x_test)]
        item_train_mse += [model_mse(item_pred, x_train)]
        item_test_mse += [model_mse(item_pred, x_test)]
    plt.figure(figsize=(8, 8))
    plt.plot(ks, user_train_mse, label='User-based train', alpha=0.5, linewidth=5)
    plt.plot(ks, user_test_mse,  label='User-based test', linewidth=5)
    plt.plot(ks, item_train_mse, label='Item-based train', alpha=0.5, linewidth=5)
    plt.plot(ks, item_test_mse,  label='Item-based test', linewidth=5)
    plt.legend(loc='best', fontsize=20)
    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('k', fontsize=30);
    plt.ylabel('MSE', fontsize=30);    
def main():
    #data_source
    data_path='ml-100k/u.data'
    #data_extract
    rating,x_train,x_test=get_data(data_path)
    #simliary
    user_sim=model_train(x_train,kind='user')
    item_sim=model_train(x_train,kind='item')
    #model_predict
    user_pre=model_predict(x_train,user_sim,kind='user')
    item_pre=model_predict(x_train,item_sim,kind='item')
    #mse
    use_mse=model_mse(user_pre,x_test)
    item_mes=model_mse(item_pre,x_test)
    #impro_model
    user_nobias=model_nobias(x_train,user_sim,kind='user')
    item_nobias=model_nobias(x_train,item_sim,kind='item')
    #top_k
    user_pred=predict_top(x_train,user_sim,kind='user',k=40)
    #item_pred=predict_top(x_train,item_sim,kind='user',k=40)
    model_plot(rating,x_train,x_test,user_sim,item_sim)
main()

