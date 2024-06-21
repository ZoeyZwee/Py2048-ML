import numpy as np

def relu(z):
    return np.maximum(z, np.zeros_like(z))

def back_relu(z):
    return np.greater(z, np.zeros_like(z))

class Layer:
    def __init__(self, size_in, size_out, fwd_fn, back_fn):
        self.shape = (size_in, size_out)
        self.x = np.zeros(size_out)
        self.b = np.zeros(size_out)
        rng = np.random.default_rng()
        self.w = rng.normal(0, np.sqrt(2/size_in), size=(size_out, size_in)) # initialize according to He distribution

        self.fwd_fn = fwd_fn
        self.back_fn = back_fn

    def apply_updates(self, weight_updates, bias_updates):
        self.w += weight_updates
        self.b += bias_updates

    def copy(self):
        """
        create a new layer with the exact same values as the current layer
        :return:  new layer
        """

        copylayer = Layer(self.shape[0], self.shape[1], self.fwd_fn, self.back_fn)
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
        self.x = self.fwd_fn(np.matmul(self.w, x_in) + self.b)
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
        dxdz = self.back_fn(z)  # derivative of activation, based on input

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

class MultiLayerPerceptron:
    """
    Feed-Forward Neural Network (MLP)
    """

    def __init__(self, shape):
        """

        :param shape: list(int). number of neurons in each layer. (including input and output layers)
        :return: MLP of specified shape with random initial weights + biases.
        """
        self.shape = shape
        self.layers = [Layer(prev, cur, relu, back_relu) for (prev, cur) in zip(shape[:-2], shape[1:-1])]
        self.layers.append(Layer(shape[-2], shape[-1], lambda x: x, lambda x: 1)) # add output layer

    def copy(self):
        """
        scores a neural network of the exact same structure with exact same weights, biases as self.
        :return: copied NN
        """
        copynn = MultiLayerPerceptron(self.shape)
        for i in range(len(self.layers)):
            copynn.layers[i] = self.layers[i].copy()
        return copynn

    def backward(self, grad_next, learning_rate):
        """
        compute desired changes for every weight, bias in network.
        :param grad_next: np-vector of upstream gradients TO each node in the output layer.
        :param learning_rate: learning rate
        """
        weight_updates = []
        bias_updates = []

        # reverse-traverse layers, backprop
        upstream = grad_next
        for i in range(len(self.layers)-1, 0, -1):  # first layer is input layer, doesn't get backpropped
            layer_wrt_w, layer_wrt_b = self.layers[i].backward(upstream, self.layers[i - 1].x)
            weight_updates.append(learning_rate * layer_wrt_w.copy())
            bias_updates.append(learning_rate * layer_wrt_b.copy())
            upstream = layer_wrt_w.transpose()
        weight_updates.reverse()
        bias_updates.reverse()
        return weight_updates, bias_updates

    def forward(self, inputs):
        """
        Update activations of all nodes in network, given activations of input layer
        :param inputs: input vector
        :return: activations of final layer
        """
        # update input layer
        self.layers[0].x = inputs
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
            self.layers[i + 1].apply_updates(delta_w[i], delta_b[i])


