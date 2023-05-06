import numpy as np

class InputLayer:
    def __init__(self, length):
        self.x = np.zeros(length)
        self.length = length

    def copy(self):
        copylayer = InputLayer(self.length)
        copylayer.x = self.x.copy()
        return copylayer

    def forward(self, x_in):
        return

    def setActivations(self, x):
        self.x = x

    def setWeights(self, w):
        return


class Layer:
    def __init__(self, x_length, b_length, w_shape):
        self.x = np.zeros(x_length)
        self.b = np.zeros(b_length)
        self.w = np.random.randn(w_shape[0], w_shape[1]) / np.sqrt(
            w_shape[1] / 2)  # each row corresponds to all weights going TO a single node of THIS layer

    def relu(self, z):
        return np.maximum(z, np.zeros_like(z))

    def back_relu(self, z):
        return np.greater(z, np.zeros_like(z))

    def update_weights_biases(self, delta_w, delta_b):
        self.w += delta_w
        self.b += delta_b

class FCLayer(Layer):
    def __init__(self, size_in, length):
        """
        Fully Connected Layer

        :param length: number of nodes in layer
        :param in_length: length of previous layer (layer to the left)
        """
        Layer.__init__(self, length, length, (length, size_in))

    def copy(self):
        """
        create a new layer with the exact same values as the current layer
        :return:  new layer
        """

        copylayer = FCLayer(self.w.shape[1], self.x.size)
        copylayer.w = self.w.copy()
        copylayer.x = self.x.copy()
        copylayer.b = self.b.copy()

        return copylayer

    def forward(self, x_in):
        """
        Updates activations of layer, given activations of previous layer and weight vector.
        :param x_in: activations of previous layer
        :return: updated activations vector
        """
        self.x = self.relu(np.matmul(self.w, x_in) + self.b)  # x = sig(W.dot(x_pre))
        return self.x

    def __backward_node(self, grad_next, x_in, w_in, b):
        """
        compute gradients of Loss function wrt weights, bias for SINGLE NODE. "downstream" gradient is wrt_weight.

        overall gradient given by: dL/dw = dL/dx * dx/dz * dz/dw
        dL/dx is precisely the upstream.                            (SCALAR)
        dx/dz is activation function (sigmoid), evaluated at z.     (SCALAR)
        dz/dw is x_in, since z=w_in*x_in + b                        (VECTOR)

        :param grad_next: np_vector containing upstream gradients to this node.
        :param x_in: np_vector of activations of previous layer (layer to the left)
        :param w_in: np_vector weights connecting this node to previous layer
        :param b: bias for this node
        :return: gradient of Loss wrt weight, bias
        """

        # compute upstream gradient
        upstream = np.sum(grad_next)

        # compute local gradient inside node
        z = np.dot(x_in, w_in) + b  # z is "sum input"
        dxdz = self.back_relu(z)  # derivative of activation, based on input

        wrt_bias = dxdz * upstream
        wrt_weight = wrt_bias * x_in

        return wrt_weight, wrt_bias

    def backward(self, grad_next, x_in):
        """
        compute gradients of Loss function wrt weights, bias for ENTIRE LAYER. "downstream" gradient is wrt_weight.

        :param grad_next: matrix of upstream gradients coming into this layer. a row corresponds to the upstreams TO a single node of THIS layer
        :param x_in: np_vector activations of previous layer (layer to the left)
        :return: [1] matrix of downstream gradients going out of this layer. a row corresponds to the downstreams TO a single node of THE PREVIOUS layer
                 [2] vector of gradients for the biases in this layer
        """

        # compute gradients for weights, bias for each node in current layer
        wrt_weight = np.zeros_like(self.w)  # matrix of same shape as weight matrix
        wrt_bias = np.zeros_like(self.b)  # vector of same length as bias vector
        for i in range(self.x.size):
            wrt_weight[i], wrt_bias[i] = self.__backward_node(grad_next[i], x_in, self.w[i], self.b[i])

        return wrt_weight, wrt_bias


class ConvLayer(Layer):
    def __init__(self, size_in, depth_in, depth, F, P):
        """
        :param size_in: size of input matrix (size_in=3 -> 3x3 input matrix)
        :param depth: number of filters
        :param F: size of filters (F=2 -> 2x2 filter)
        :param P: amount of zero-padding

        each row of weight matrix corresponds to a filter
        """
        # TODO: depth
        Layer.__init__(self, size_in ** 2, depth, (depth, F ** 2))
        self.size_in = size_in
        self.depth = depth
        self.depth_in = depth_in
        self.F = F
        self.P = P

    def __get_regions(self, x_in):
        """
        Take input vector and return a list of the (flattened) sub-matrices for convolution
        :param M: input matrix
        :return: column vectors for dot products
        """
        # TODO: depth
        M = x_in.reshape(self.size_in, self.size_in)
        cols = []
        height, width = np.shape(M)
        top_pad = np.zeros((self.P, width + 2 * self.P))
        mid = np.hstack((np.zeros((height, self.P)), M, np.zeros((height, self.P))))
        bot_pad = np.zeros((self.P, width + 2 * self.P))
        M = np.vstack((top_pad, mid, bot_pad))

        for i in range(height - 2 * self.P):
            for j in range(width - 2):
                cols.append(M[i:i + self.F, j:j + self.F].flatten().transpose())
        return cols

    def forward(self, x_in):
        """
        compute and update activations of current layer, given activations of previous layer
        :param x_in: activations of previous layer
        :return: activations of current layer
        """
        # TODO: depth
        conv_regions = self.__get_regions(x_in)
        z = np.zeros(len(conv_regions))  # post convolution, pre relu
        for col in conv_regions:
            for i in range(self.depth):
                z[i] += np.dot(self.w[i], col[i])
        z = np.array(z)
        return self.relu(z)

    def backward(self, grad_in, x_in):
        """
        compute gradient for all regions wrt weights, biases
        :param grad_in: upstream gradients. each row corresponds to upstreams TO a node of THIS layer
        :param x_in: activations of previous layer
        :return: grad wrt weights, biases
        """

        conv_regions = self.__get_regions(x_in)
        for i in range(len(conv_regions)):
            dxdw_i = self.__backward_node(conv_regions[i])

    def __backward_node(self, x_region):
        """
        compute local gradient at this node
        :param x_region: activations of relevant region
        :return: local gradient (dxdw)
        """
        # dxdz = back_relu(self.z) = back_relu(self.x)
        # dzdw = x_region
        return self.back_relu(self.x) * x_region


class Network:
    """
    for now network is ALWAYS a sigmoid network.
    """

    def __init__(self, nn_architecture):
        self.layers = []

        """ example architecture
            [
                {"length_input": 0, "length": 4, "activation": "input"},

                {"length_input": 4, "length": 16, "activation": "sigmoid"},
                {"length_input": 16, "length": 16, "activation": "sigmoid"},
                {"length_input": 16, "length": 16, "activation": "sigmoid"},
                {"length_input": 16, "length": 16, "activation": "sigmoid"},

                {"length_input": 16, "length": 4, "activation": "sigmoid"},
            ]
        """
        self.nn_architecture = nn_architecture

        # init layers
        for entry in nn_architecture:
            if entry["activation"] == "input":
                layer = InputLayer(entry["length_input"])
            elif entry["activation"] == "sigmoid":
                layer = FCLayer(entry["length_input"], entry["length"])
            elif entry["activation"] == "relu":
                layer = FCLayer(entry["length_input"], entry["length"])
            else:
                layer = FCLayer(entry["length_input"], entry["length"])
            self.layers.append(layer)

    def copy(self):
        """
        scores a neural network of the exact same structure with exact same weights, biases as self.
        :return: copied NN
        """
        copynn = Network(self.nn_architecture)
        for i in range(len(self.layers)):
            copynn.layers[i] = self.layers[i].copy()
        return copynn

    def backward(self, grad_next, alpha):
        """"
        compute desired changes for every weight, bias in network.
        :param grad_next: np-vector of upstream gradients TO each node in the output layer.
        :param alpha: learning rate
        """
        delta_w = []
        delta_b = []

        # reverse-traverse layers, backprop
        upstream = grad_next
        for i in range(len(self.layers) - 1, 0, -1):  # first layer is input layer, doesn't get backpropped
            layer_wrt_w, layer_wrt_b = self.layers[i].backward(upstream, self.layers[i - 1].x)
            delta_w.insert(0, alpha * layer_wrt_w.copy())
            delta_b.insert(0, alpha * layer_wrt_b.copy())
            upstream = layer_wrt_w.transpose()
        return delta_w, delta_b

    def forward(self, inputs):
        """
        Update activations of all nodes in network, given activations of input layer
        :param inputs: input vector
        :return: activations of final layer
        """
        # update input layer
        self.layers[0].setActivations(inputs)
        x_in = None  # activations of previous layer
        for layer in self.layers:
            if x_in is None:
                x_in = inputs
            layer.forward(x_in)
            x_in = layer.x

        return x_in  # will be activations of final layer, which is big happy :)

    def update_weights_biases(self, delta_w, delta_b):
        """
        Update weights and biases of network in accordance to desired changes
        :param delta_w: plist - each entry corresponds to a layer
        :param delta_b: plist - each entry corresponds to a layer
        :return: None
        """

        for i in range(len(delta_w)):
            self.layers[i + 1].update_weights_biases(delta_w[i], delta_b[i])


