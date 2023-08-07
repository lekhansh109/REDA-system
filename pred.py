from Reda import Reda
from RedaData import RedaData

from configparser import ConfigParser
config = ConfigParser()
config.read('conf', encoding='UTF-8')

if __name__ == '__main__':
    mr_model = Reda()
    #mr_model.model_train()
    #mr_model.model_test()
    #user = int(input("Enter User No.: "))
    list_user = [int(item) for item in input("Enter the list items : ").split()]
    dirname = config['DEFAULT'].get("dirname")
    data = RedaData(dirname)
    itemset, userset = data.getItems()
    print(len(itemset), len(userset))
    ll = set(list_user)
    target = list(itemset - ll)

        
    score = mr_model.prediction(list_user, target)

    #print(result)
    


