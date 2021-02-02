import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ReplicationPad3d


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


class I3D(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, inp):
        # Preprocessing
        inp = torch.cat([inp, inp, inp], 3)
        inp = torch.cat([inp, inp, inp, inp, inp, inp], 2)
        print("input: ", inp.size())
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        print("mixed_5c: ", out.size())
        # out = self.avg_pool(out)
        out = F.adaptive_avg_pool3d(out, (None, 1, 1))
        print("avg_pool: ", out.size())
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        print("conv3d: ", out.size())
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        print("logits: ", out.size())
        out_logits = out
        out = self.softmax(out_logits)
        return out, out_logits

    def extract(self, inp):
        print("input: ", inp.size())
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        # print "mixed_5c: ", out.size()
        out = F.adaptive_avg_pool3d(out, (None, 1, 1))
        # print "avg_pool: ", out.size()
        out = out.squeeze(3)
        out = out.squeeze(3)

        return out.transpose(1, 2)

    def load_tf_weights(self, sess):
        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        elif self.modality == 'flow':
            prefix = 'Flow/inception_i3d'
        load_conv3d(state_dict, 'conv3d_1a_7x7', sess,
                    os.path.join(prefix, 'Conv3d_1a_7x7'))
        load_conv3d(state_dict, 'conv3d_2b_1x1', sess,
                    os.path.join(prefix, 'Conv3d_2b_1x1'))
        load_conv3d(state_dict, 'conv3d_2c_3x3', sess,
                    os.path.join(prefix, 'Conv3d_2c_3x3'))

        load_mixed(state_dict, 'mixed_3b', sess,
                   os.path.join(prefix, 'Mixed_3b'))
        load_mixed(state_dict, 'mixed_3c', sess,
                   os.path.join(prefix, 'Mixed_3c'))
        load_mixed(state_dict, 'mixed_4b', sess,
                   os.path.join(prefix, 'Mixed_4b'))
        load_mixed(state_dict, 'mixed_4c', sess,
                   os.path.join(prefix, 'Mixed_4c'))
        load_mixed(state_dict, 'mixed_4d', sess,
                   os.path.join(prefix, 'Mixed_4d'))
        load_mixed(state_dict, 'mixed_4e', sess,
                   os.path.join(prefix, 'Mixed_4e'))
        # Here goest to 0.1 max error with tf
        load_mixed(state_dict, 'mixed_4f', sess,
                   os.path.join(prefix, 'Mixed_4f'))

        load_mixed(
            state_dict,
            'mixed_5b',
            sess,
            os.path.join(prefix, 'Mixed_5b'),
            fix_typo=True)
        load_mixed(state_dict, 'mixed_5c', sess,
                   os.path.join(prefix, 'Mixed_5c'))
        load_conv3d(
            state_dict,
            'conv3d_0c_1x1',
            sess,
            os.path.join(prefix, 'Logits', 'Conv3d_0c_1x1'),
            bias=True,
            bn=False)
        self.load_state_dict(state_dict)


def get_conv_params(sess, name, bias=False):
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'w:0'))
    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(
            os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
    conv_weights = sess.run(conv_weights_tensor)
    conv_shape = conv_weights.shape

    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(
        os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]
    conv_params = [
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding
    ]
    if bias:
        conv_params.append(conv_bias)
    return conv_params


def get_bn_params(sess, name):
    moving_mean_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(
        os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)
    return moving_mean, moving_var, beta


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == "VALID":
        return [0, 0]
    elif padding_name == "SAME":
        # return [math.ceil(int(conv_shape[0])/2), math.ceil(int(conv_shape[1])/2)]
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        raise ValueError('Invalid padding name ' + padding_name)


def load_conv3d(state_dict, name_pt, sess, name_tf, bias=False, bn=True):
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    conv_weights_rs = np.transpose(
        conv_weights, (4, 3, 0, 1,
                       2))  # to pt format (out_c, in_c, depth, height, width)
    state_dict[name_pt + '.conv3d.weight'] = torch.from_numpy(conv_weights_rs)
    if bias:
        state_dict[name_pt + '.conv3d.bias'] = torch.from_numpy(conv_bias)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = torch.ones(out_planes)
        state_dict[name_pt +
                   '.batch3d.bias'] = torch.from_numpy(beta.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_mean'] = torch.from_numpy(moving_mean.squeeze())
        state_dict[name_pt
                   + '.batch3d.running_var'] = torch.from_numpy(moving_var.squeeze())


def load_mixed(state_dict, name_pt, sess, name_tf, fix_typo=False):
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess,
                os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess,
                os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess,
                os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess,
                    os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess,
                os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))

#
# these are labels from kinetics dataset in order of which the model was trained
# ideally this should be in a seperate JSON or YAML file, but I am writing them
# here for convinience.
#
LABELS = [
  'abseiling', 'air drumming', 'answering questions', 'applauding', 'applying cream',
  'archery', 'arm wrestling', 'arranging flowers', 'assembling computer', 'auctioning',
  'baby waking up', 'baking cookies', 'balloon blowing', 'bandaging', 'barbequing',
  'bartending', 'beatboxing', 'bee keeping', 'belly dancing', 'bench pressing',
  'bending back', 'bending metal', 'biking through snow', 'blasting sand', 'blowing glass',
  'blowing leaves', 'blowing nose', 'blowing out candles', 'bobsledding', 'bookbinding',
  'bouncing on trampoline', 'bowling', 'braiding hair', 'breading or breadcrumbing',
  'breakdancing', 'brush painting', 'brushing hair', 'brushing teeth', 'building cabinet',
  'building shed', 'bungee jumping', 'busking', 'canoeing or kayaking', 'capoeira',
  'carrying baby', 'cartwheeling', 'carving pumpkin', 'catching fish',
  'catching or throwing baseball', 'catching or throwing frisbee',
  'catching or throwing softball', 'celebrating', 'changing oil', 'changing wheel',
  'checking tires', 'cheerleading', 'chopping wood', 'clapping', 'clay pottery making',
  'clean and jerk', 'cleaning floor', 'cleaning gutters', 'cleaning pool', 'cleaning shoes',
  'cleaning toilet', 'cleaning windows', 'climbing a rope', 'climbing ladder', 'climbing tree',
  'contact juggling', 'cooking chicken', 'cooking egg', 'cooking on campfire',
  'cooking sausages', 'counting money', 'country line dancing', 'cracking neck',
  'crawling baby', 'crossing river', 'crying', 'curling hair', 'cutting nails',
  'cutting pineapple', 'cutting watermelon', 'dancing ballet', 'dancing charleston',
  'dancing gangnam style', 'dancing macarena', 'deadlifting', 'decorating the christmas tree',
  'digging', 'dining', 'disc golfing', 'diving cliff', 'dodgeball', 'doing aerobics',
  'doing laundry', 'doing nails', 'drawing', 'dribbling basketball', 'drinking',
  'drinking beer', 'drinking shots', 'driving car', 'driving tractor', 'drop kicking',
  'drumming fingers', 'dunking basketball', 'dying hair', 'eating burger', 'eating cake',
  'eating carrots', 'eating chips', 'eating doughnuts', 'eating hotdog', 'eating ice cream',
  'eating spaghetti', 'eating watermelon', 'egg hunting', 'exercising arm',
  'exercising with an exercise ball', 'extinguishing fire', 'faceplanting', 'feeding birds',
  'feeding fish', 'feeding goats', 'filling eyebrows', 'finger snapping', 'fixing hair',
  'flipping pancake', 'flying kite', 'folding clothes', 'folding napkins', 'folding paper',
  'front raises', 'frying vegetables', 'garbage collecting', 'gargling', 'getting a haircut',
  'getting a tattoo', 'giving or receiving award', 'golf chipping', 'golf driving',
  'golf putting', 'grinding meat', 'grooming dog', 'grooming horse', 'gymnastics tumbling',
  'hammer throw', 'headbanging', 'headbutting', 'high jump', 'high kick', 'hitting baseball',
  'hockey stop', 'holding snake', 'hopscotch', 'hoverboarding', 'hugging', 'hula hooping',
  'hurdling', 'hurling (sport)', 'ice climbing', 'ice fishing', 'ice skating', 'ironing',
  'javelin throw', 'jetskiing', 'jogging', 'juggling balls', 'juggling fire',
  'juggling soccer ball', 'jumping into pool', 'jumpstyle dancing', 'kicking field goal',
  'kicking soccer ball', 'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing',
  'laying bricks', 'long jump', 'lunge', 'making a cake', 'making a sandwich', 'making bed',
  'making jewelry', 'making pizza', 'making snowman', 'making sushi', 'making tea', 'marching',
  'massaging back', 'massaging feet', 'massaging legs', "massaging person's head",
  'milking cow', 'mopping floor', 'motorcycling', 'moving furniture', 'mowing lawn',
  'news anchoring', 'opening bottle', 'opening present', 'paragliding', 'parasailing',
  'parkour', 'passing American football (in game)', 'passing American football (not in game)',
  'peeling apples', 'peeling potatoes', 'petting animal (not cat)', 'petting cat',
  'picking fruit', 'planting trees', 'plastering', 'playing accordion', 'playing badminton',
  'playing bagpipes', 'playing basketball', 'playing bass guitar', 'playing cards',
  'playing cello', 'playing chess', 'playing clarinet', 'playing controller', 'playing cricket',
  'playing cymbals', 'playing didgeridoo', 'playing drums', 'playing flute', 'playing guitar',
  'playing harmonica', 'playing harp', 'playing ice hockey', 'playing keyboard',
  'playing kickball', 'playing monopoly', 'playing organ', 'playing paintball',
  'playing piano', 'playing poker', 'playing recorder', 'playing saxophone',
  'playing squash or racquetball', 'playing tennis', 'playing trombone', 'playing trumpet',
  'playing ukulele', 'playing violin', 'playing volleyball', 'playing xylophone', 'pole vault',
  'presenting weather forecast', 'pull ups', 'pumping fist', 'pumping gas', 'punching bag',
  'punching person (boxing)', 'push up', 'pushing car', 'pushing cart', 'pushing wheelchair',
  'reading book', 'reading newspaper', 'recording music', 'riding a bike', 'riding camel',
  'riding elephant', 'riding mechanical bull', 'riding mountain bike', 'riding mule',
  'riding or walking with horse', 'riding scooter', 'riding unicycle', 'ripping paper',
  'robot dancing', 'rock climbing', 'rock scissors paper', 'roller skating',
  'running on treadmill', 'sailing', 'salsa dancing', 'sanding floor', 'scrambling eggs',
  'scuba diving', 'setting table', 'shaking hands', 'shaking head', 'sharpening knives',
  'sharpening pencil', 'shaving head', 'shaving legs', 'shearing sheep', 'shining shoes',
  'shooting basketball', 'shooting goal (soccer)', 'shot put', 'shoveling snow',
  'shredding paper', 'shuffling cards', 'side kick', 'sign language interpreting', 'singing',
  'situp', 'skateboarding', 'ski jumping', 'skiing (not slalom or crosscountry)',
  'skiing crosscountry', 'skiing slalom', 'skipping rope', 'skydiving', 'slacklining',
  'slapping', 'sled dog racing', 'smoking', 'smoking hookah', 'snatch weight lifting',
  'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling',
  'somersaulting', 'spinning poi', 'spray painting', 'spraying', 'springboard diving',
  'squat', 'sticking tongue out', 'stomping grapes', 'stretching arm', 'stretching leg',
  'strumming guitar', 'surfing crowd', 'surfing water', 'sweeping floor', 'swimming backstroke',
  'swimming breast stroke', 'swimming butterfly stroke', 'swing dancing', 'swinging legs',
  'swinging on something', 'sword fighting', 'tai chi', 'taking a shower', 'tango dancing',
  'tap dancing', 'tapping guitar', 'tapping pen', 'tasting beer', 'tasting food', 'testifying',
  'texting', 'throwing axe', 'throwing ball', 'throwing discus', 'tickling', 'tobogganing',
  'tossing coin', 'tossing salad', 'training dog', 'trapezing', 'trimming or shaving beard',
  'trimming trees', 'triple jump', 'tying bow tie', 'tying knot (not on a tie)', 'tying tie',
  'unboxing', 'unloading truck', 'using computer', 'using remote controller (not gaming)',
  'using segway', 'vault', 'waiting in line', 'walking the dog', 'washing dishes',
  'washing feet', 'washing hair', 'washing hands', 'water skiing', 'water sliding',
  'watering plants', 'waxing back', 'waxing chest', 'waxing eyebrows', 'waxing legs',
  'weaving basket', 'welding', 'whistling', 'windsurfing', 'wrapping present', 'wrestling',
  'writing', 'yawning', 'yoga', 'zumba']
