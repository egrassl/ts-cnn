import argparse
import yaml


# Creates arg parsers
parser = argparse.ArgumentParser()

parser.add_argument('name', type=str, metavar='model_name', help='Name that will be used in this model')
parser.add_argument('type', type=str, metavar='stream_type', choices=['s', 't', 'st'], help='CNN stream type')
parser.add_argument('config', type=str, metavar='config_file', help='Config file path')
parser.add_argument('--gpu', type=int, metavar='gpu_id', help='Specifies a GPU to run the application')

args = parser.parse_args()

# Gets parameters
with open(args.config) as c_file:
    configs = yaml.load(c_file, Loader=yaml.FullLoader)

    # keras imports
    from models.vgg16 import VGG16
    from keras_extensions import custom_crops

    # Loads cnn model
    if args.type == 's':
        input_shape = (224, 224, 3)

    elif args.type == 't':
        input_shape = (224, 224, 20)

    else:
        raise NotImplementedError()


model = VGG16(
    args.name,
    input_shape=input_shape,
    classes=101,
    dropout=configs['dropout'],
    l2_reg=configs['l2_reg'],
    weights=configs['weights'],
    include_top=True
)

model.summary()
