# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import util
import generate_train_set
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # print_hi('PyCharm')
   #
   #  #test_import_file
   #  list_file=utils.load_file("./text.txt")
   #  #test_output_file
   #  text_embeding_list=[[1,3,4],[2,3,4],[4,5,6]]
   #  print(list(range(10)) + [10, 11, 12])
   #  utils.save_file(text_embeding_list,"./text2.txt")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
    #test

   #test generate_hrt
    entity_path="./1_entity.txt"
    relation_path="./1_relation.txt"
    pool_path="./1_pool.txt"
    hist_path="./1_hist.txt"
    attr_path = "./1_attr.txt"
    htr=generate_train_set.generate_htr(entity_path, relation_path, pool_path, None, attr_path)
    # print(en1)
    # print("*************8")
    # print(en2)
    print(htr[0])
    print(htr[1])