# DFPSystem(基于OpenCV和卷积神经网络的预防驾驶疲劳系统)

 <div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.0-green)
![CUDA](https://img.shields.io/badge/CUDA-11.0-green)
![cuDNN](https://img.shields.io/badge/cuDNN-8.0-green)
![Python](https://img.shields.io/badge/Python-3.7.4-green)

</div>


所使用的数据集来自
http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html
http://vis-www.cs.umass.edu/lfw/

## 处理数据集

请保证环境配置正确，以便接下来的运行。具体的第三方库请自行查找。

请在文件夹下创建两个新文件夹 -> data & output

在data文件夹下 新建五个文件夹 closed_eyes_face_data & open_eyes_face_data & gridsearches & result & train

运行 closed_eyes_cropper.py 以及 open_eyes_cropper.py 以从数据集中裁剪图片

## 网格搜索

运行 importer.py 获取网格搜索结果

## 训练模型

运行 ModelExport.py 训练模型并保存至文件夹下

为了更好地得到结果，请改变 ModelExport.py 最后一行 best_model_5.h5 -> best_model_x.h5 再运行

## 实现实时监测

确保你的显卡显存足够支撑，否则很容易卡死

运行 run.py 即刻

## 相关问题

关于 SpeedTest.py 使用其中一个数据来验证模型准确度

数据集若下载失败，请科学上网，或者直接提issue 改天上传

如果你在模型训练的时候报 Out of memory 错误，请修改 batch_size 小一点，不过会影响模型

禁止用于商业用途，如有侵权请联系我进行删除，若有其他问题请直接提issue

## 鸣谢

towrads data science Author:Dustin Stewart

如果对你有帮助，请点个Star!!!!!
