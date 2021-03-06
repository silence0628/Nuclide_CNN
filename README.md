python 环境中运行，进入当前项目所在文件目录

> > python main.py --epoch 50
> >                   --batch_size 128
> >                   --lr 0.001
> >                   --phase feature_extraction
> >                   --train_dir
> >                   --iteration
> >                   --feature_dir

本项目可以直接拷贝到任意目录下进行运行，
    需要注意的是 train_dir 需要就行相应的修改：如  --train_dir /xxx/xxx/train/
    不修改对应参数即可认为 选择默认的对应参数



model.py
    采用基本的2-D CNN模型，具体为
    conv1 --> pool1 --> conv2 --> pool2 --> local3 --> local4 --> softmax

相关命令
训练
    1 以　2Ｄ图像能谱数据进行训练
      >> python main.py --phase train --inference 2D --checkpoint ./checkpoint/

```
2 以　1Ｄ能谱数据进行训练
  >> python main.py --phase train --inference 1D --checkpoint ./checkpoint_1D/
```

特征提取
默认参数：以2D数据为主，默认进行２Ｄ数据特征提取，选择训练好的２Ｄ模型，默认保存数据
    1 以　2Ｄ图像能谱数据进行网络层特征提取，并进行保存
      >> python main.py --phase feature_extraction 　　　　　　　　#　进行特征提取
      　　　　　　　　　--inference 2D --checkpoint ./checkpoint/  #  2D模型地址设置
      　　　　　　　　　--isSaveFeature True　　　　　　　　　　　 #  是否进行特征向量保存　True for Save, xxx for not save

```
2 以　1Ｄ能谱数据进行网络层特征提取
  >> python main.py --phase feature_extraction 　　　　　　　　   #　进行特征提取
  　　　　　　　　　--inference 1D --checkpoint ./checkpoint_1D/  #  1D模型地址设置
  　　　　　　　　　--isSaveFeature xxxx　　　　　　　　　　　    #  是否进行特征向量保存　True for Save, xxx for not save
```