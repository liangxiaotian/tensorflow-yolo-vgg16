common_params = {'image_size': '224',
                 'batch_size': '8',
                 'num_classes': '20',
                 'max_objects_per_image':'20'}

dataset_params = {'name':'yolo.dataset.text_dataset.TextDataset',
                  'path':'data/pascal_voc.txt',
                  'thread_num':'3'}

net_params = {'name': 'yolo.net.vgg16.Vgg16',
            'weight_decay':'0.0005',
            'cell_size': '7',
            'boxes_per_cell': '2',
            'object_scale':'1',
            'noobject_scale':'0.5',
            'class_scale':'1',
            'coord_scale':'5'}
solver_params = {'name': 'yolo.solver.yolo_solver.YoloSolver',
            'learning_rate': '0.000001',
            'moment': '0.9',
            'max_iterators': '1000000',
            'pretrain_model_path': 'models/pretrain/vgg_16.ckpt',
            'train_dir': 'models/train'
}

def get_params():
    return common_params, dataset_params, net_params, solver_params