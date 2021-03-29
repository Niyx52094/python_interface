# python_interface
*
* 那些“1_”开头的是根据要求写的一些测试文件，包括```entity```,```relation```,```pool```,```attr```,```histoty```
* 
* 其他的放在```Recommend/data/check_data```里面的文件是```all_relation_and IDS``` 和 ```attr_entity_IDS_start_from XX``` 是调用```generate_train_set.py```后生成的ID文件，用来做校验
* ```new_item_id_with_original_item_id.txt``` 和```new_user_id_with_original_id.txt```是把输入的id转换成都从0开始的新id。这里用来后续做对照

## 流程
* 打开```rec_main.py``` 运行，里面有输入五个文件的路径。可以把那五个文件的路径放进去。直接运行她就会自己跑完，但是最后一步输出还没写。//03-29,update：输出已写完
* 输出的文件在```/Recommend/data/output_data```中。输出两个文件，每个文件entity的数目都和输入```entity.txt```一致，就是说attr的entity没有加进来的，具体描述在下面。


## 进度
* 流程是先```generate_train_set.py```输入1_开头的五个文件，生成```user```,```item```,```attr```,```relation```的(h,t,r)图文件，总共会生成两个文件。
  * 一个文件形式为```(h,positive_item,netative_item,relation)``` 输入到```pretran_train_transD.py```预训练TransD的模型，再train的时候会用。
  * 另一个文件形式需要把蕴文没加的```relation```加进来，用来输入```./Module.py```里面生成最终的embedding。目前正在做这一块。//03-28,update:已做完
### 2021-03-28
* Module.py 修改完毕
* debug完毕，程序跑通，但是用的我自己的测试文件。。仅仅几个例子。最后的loss算出来都是0，不确定是样本少所以没损失还是哪里出了问题
* TODO:保存跑完的model并输出最后的embedding.txt
### 2021-03-29
* 输出代码完毕，总共会输出两个txt：
 * ```embedding_1.txt```是输出训练之后的transD里的embedding,维度64位。
 * ```embedding_2.txt```是输出上述的embedding经过一个三层NN,每一层输出一个embedding就和之前的concat。这么做是把该节点邻居节点的信息也加上，所以维度是64X4。
 
