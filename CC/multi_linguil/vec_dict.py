import pickle
root = "/root/data/xiaoyang/PyTorch-CycleGAN-master/cls-acl10-unprocessed"
languages = ['de', 'en', 'fr', 'jp']
for lan in languages:
    print(lan)
    f = open("/root/data/wiki.multi."+lan+".vec", 'r')
    language_dict = {}
    line = f.readline()
    while line:
        try:
            line = f.readline()
            if not line:
                break
            line = line.split()
            key = line[0]
            vec = [float(element) for element in line[1:]]
            language_dict[key] = vec
        except Exception as e:
            print(e)
    f.close()
    f = open("/root/data/"+lan+".pkl", 'wb+')
    pickle.dump(language_dict, f)
    f.close()
