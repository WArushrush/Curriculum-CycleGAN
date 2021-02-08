import os,torch,pickle
root="/rscratch/schzhao/xiaoyang/mtl_dataset"
product=["apparel","baby","books","camera","dvd","electronics","health_personal_care",
         "imdb","kitchen","magazines","MR","music","software","sports","toys_games","video"]
train_dir=[root+"/"+pro+"_train.pkl" for pro in product]
text_dir=[root+"/"+pro+"_test.pkl" for pro in product]
