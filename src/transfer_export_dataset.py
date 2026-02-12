# transfer_export_dataset.py
# expected directory tree:
#├── kaggle
#│   ├── transfer_train
#│   |   ├── class1
#│   |   ├── class2
#│   |   └── ...
#│   └── transfer_test
#│   |   ├── class1
#│   |   ├── class2
#│   |   └── ...

from config import transfer_train_directory, transfer_test_directory, transfer_train_size, transfer_categories
import os
import shutil

def copy_dataset(source_directory, target_directory, categories=None, target_size=None):
    if categories is None:
        categories = os.listdir(source_directory)
    
    for category in categories:
        source_category_dir = os.path.join(source_directory, category)
        target_category_dir = os.path.join(target_directory, category)
        
        if not os.path.exists(target_category_dir):
            os.makedirs(target_category_dir)
        
        image_files = os.listdir(source_category_dir)
        if target_size is None:
            count = len(image_files)
        else:
            count = int(target_size/len(categories))
            if category == categories[-1]:
                count += target_size%len(categories)
        
        for image_file in image_files[:count]:
            source_image_path = os.path.join(source_category_dir, image_file)
            target_image_path = os.path.join(target_category_dir, image_file)
            shutil.copy(source_image_path, target_image_path)


if __name__ == "__main__":
    source_directory = 'kaggle'
    train_source_directory = os.path.join(source_directory, 'transfer_train')
    test_source_directory = os.path.join(source_directory, 'transfer_test')
    train_target_directory = transfer_train_directory
    test_target_directory = transfer_test_directory

    copy_dataset(train_source_directory, train_target_directory, transfer_categories, transfer_train_size)
    print('Transfer train dataset have been extracted.')
    
    copy_dataset(test_source_directory, test_target_directory, transfer_categories, None)
    print('Transfer test dataset have been extracted.')