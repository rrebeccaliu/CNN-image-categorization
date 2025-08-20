from torch import nn


def cnn_categorization_improved(netspec_opts):
    """
    Constructs a network for the improved categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the improved network's architecture.

    Returns
    -------
    A categorization model which can be trained by PyTorch

    """

    net = nn.Sequential()
    kernel_size_l = netspec_opts["kernel_size"]
    num_filters_l = netspec_opts["num_filters"]
    stride_l = netspec_opts["stride"]
    layer_type_l = netspec_opts["layer_type"]
    L = len(kernel_size_l)
    in_channels = 3
    # add layers as specified in netspec_opts to the network
    for i in range(L):
        if layer_type_l[i] == 'conv':
            k = kernel_size_l[i]
            if isinstance(k , int):
                padding =( k - 1 ) // 2
            else:
                paddingx = (k[0] - 1) // 2
                paddingy = (k[1] - 1) // 2
                padding = (paddingx, paddingy)

            
            num_filters = num_filters_l[i]
            
            net.add_module(f'conv{i}', nn.Conv2d(in_channels, num_filters, kernel_size_l[i], stride_l[i], padding))

           # print(f'conv{i}', in_channels, num_filters, kernel_size_l[i], stride_l[i], padding)

            in_channels = num_filters

        elif layer_type_l[i] == 'bn':
            net.add_module(f'bn{i}', nn.BatchNorm2d(num_filters_l[i]))
            #print(f'bn{i}',num_filters_l[i])


        elif layer_type_l[i] == "relu":
            net.add_module(f'relu{i}', nn.ReLU())
           # print(f'relu{i}')
        
        elif layer_type_l[i] == "pool":
            net.add_module(f'pool{i}', nn.AvgPool2d(kernel_size_l[i], stride_l[i]))
           # print(f'pool{i}', kernel_size_l[i], stride_l[i])


    #net.add_module('fully connected', nn.Linear(num_filters_l[-1], 16))
    return net
