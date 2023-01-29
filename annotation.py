import os
import fnmatch


def wrgbd_annotation(path):
    path_suffix = '*_crop.png'
    data_path = path
    categories = []
    with open('testinstance_ids.txt', mode='r') as f:
        test_instance = f.readlines()
        test_instance = [i.strip() for i in test_instance]

        trial1 = test_instance[1:52]
        trial2 = test_instance[55:106]
        trial3 = test_instance[109:160]
        trial4 = test_instance[163:214]
        trial5 = test_instance[217:268]
        trial6 = test_instance[271:322]
        trial7 = test_instance[325:376]
        trial8 = test_instance[379:430]
        trial9 = test_instance[433:484]
        trial10 = test_instance[487:538]
        f.close()
    trials = [trial1, trial2, trial3, trial4, trial5, trial6, trial7, trial8, trial9, trial10]
    for i, trial in enumerate(trials, start=1):
        test_file = open('trial_' + str(i) + '_test.txt', 'w')
        train_file = open('trial_' + str(i) + '_train.txt', 'w')
        for idx, category in enumerate(sorted(os.listdir(data_path))):
            category_path = os.path.join(data_path, category)
            categories.append(category)
            for instance in sorted(os.listdir(category_path)):

                if instance in trial:
                    instance_path = os.path.join(category_path, instance)
                    num_img = len(fnmatch.filter(sorted(os.listdir(instance_path)), path_suffix))
                    for img in fnmatch.filter(sorted(os.listdir(instance_path)), path_suffix):
                        fram = int(img.split('_')[3])
                        counter = int(img.split('_')[3][-1])
                        if counter == 1 or counter == 6 or fram == num_img:
                            img_path = os.path.join(instance_path, img)
                            test_file.write(img_path + ' ' + str(idx))
                            test_file.write('\n')
                            counter += 5
                else:
                    instance_path = os.path.join(category_path, instance)
                    num_img = len(fnmatch.filter(sorted(os.listdir(instance_path)), path_suffix))
                    for img in fnmatch.filter(sorted(os.listdir(instance_path)), path_suffix):
                        fram = int(img.split('_')[3][-1])
                        counter = int(img.split('_')[3])
                        if fram == 1 or fram == 6 or counter == num_img:
                            img_path = os.path.join(instance_path, img)
                            train_file.write(img_path + ' ' + str(idx))
                            train_file.write('\n')

        test_file.close()
        train_file.close()