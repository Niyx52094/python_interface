# python_interface
*
* 那些“1_”开头的是根据要求写的一些测试文件，包括```entity```,```relation```,```pool```,```attr```,```histoty```
* 
* 其他的文件```all_relation_and IDS``` 和 ```attr_entity_IDS_start_from XX``` 是调用```generate_train_set.py```后生成的ID文件，用来做校验
* 
* ```new_item_id_with_original_item_id.txt``` 和```new_user_id_with_original_id.txt```是把输入的id转换成都从0开始的新id。这里用来后续做对照
* 
* 流程是用```generate_train_set.py```输入1_开头的五个文件，生成```user```,```item```,```attr```,```relation```的(h,t,r)图文件，总共会生成两个文件。
  * 一个文件形式为```(h,positive_item,netative_item,relation)``` 输入到```pretran_train_transD.py```预训练。
  * 另一个文件形式需要把蕴文没加的```relation```加进来，用来输入```./Recommend/Module.py```里面生成最终的embedding。目前正在做这一块。

 
