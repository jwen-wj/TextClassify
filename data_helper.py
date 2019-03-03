import jieba
import codecs
import re
import numpy as np
import csv
import importlib,sys
importlib.reload(sys)

class TextConfig():
    originTrainData = './OriginData/train.tsv'
    originTestData = './OriginData/test.tsv'

    trainFileCsv = './OriginData/handledTrain.csv'

    trainFile = './OriginData/handledTrain.txt'
    testFile = './OriginData/test.txt'

#
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    生成一个batch迭代器
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]



def HandleOriginData():
    '''
    处理原始数据
    存入新文件中
    '''
    with open(TextConfig.originTrainData) as file:
        reader = csv.reader(file)
        next(reader)
        with codecs.open(TextConfig.trainFileCsv,"w",'utf_8_sig') as writeFile:
            write = csv.writer(writeFile)
            for row in reader:
                rowContent = (row[0]).split("\t")
                if(len(rowContent)<2):
                    continue
                originLine = rowContent[0]
                originTags = rowContent[1]
                line = jieba.lcut(ClearUselessWord(originLine))
                tags = originTags.split("--")
                write.writerow([(" ".join(line)),(" ".join(tags))])



def ClearUselessWord(line):
    """
        1. 将除汉字外的字符转为一个空格
        2. 将连续的多个空格转为一个空格
        3. 除去句子前后的空格字符
        """
    line = re.sub(r'[^\u4e00-\u9fffQ]', ' ', line)
    line = re.sub(r'\s{2,}', ' ', line)
    return line.strip()



if __name__ == '__main__':

    # HandleOriginData()
    print("data_helper完成")



