class NN {
    constructor(no_inp_nodes, no_ans_nodes, no_hidden_layers, no_hidden_nodes, activation, dactivation) {
        this.no_inp_nodes = no_inp_nodes;
        this.no_ans_nodes = no_ans_nodes;
        this.no_hidden_layers = no_hidden_layers;
        this.no_hidden_nodes = no_hidden_nodes;

        this.layers;
        this.weights;
        this.bias;
        this.forw_prop;

        this.activation = activation;
        this.dactivation = dactivation;

        this.learning_rate;
    }

    initialise() {
        let temp;

        this.layers = [];
        temp = new Matrix(this.no_inp_nodes, 1);
        this.layers.push(temp);
        for (let i = 0; i < this.no_hidden_layers; i++) {
            temp = new Matrix(this.no_hidden_nodes[i], 1);
            this.layers.push(temp);
        }
        temp = new Matrix(this.no_ans_nodes, 1);
        this.layers.push(temp);

        this.weights = [];
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.weights.push(new Matrix(this.layers[i + 1].rows, this.layers[i].rows));
        }

        this.bias = [];
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.bias.push(new Matrix(this.layers[i + 1].rows, 1));
        }
    }

    forwardPropagation(inputs_matrix) {
        this.forw_prop = [inputs_matrix];
        for (let i = 0; i < this.bias.length; i++) {
            this.forw_prop.push(Matrix.multiply(this.weights[i], this.forw_prop[i]));
            this.forw_prop[i + 1] = Matrix.add(this.forw_prop[i + 1], this.bias[i]);
            this.forw_prop[i + 1].map(this.activation);
        }
        return (this.forw_prop[this.forw_prop.length - 1]);
    }

    backPropagation(output_matrix, _learning_rate) {
        this.learning_rate = _learning_rate;

        // Calculate the weighted errors
        let weighted_errors = [Matrix.subtract(output_matrix, this.forw_prop[this.forw_prop.length - 1])];
        for (let i = 1; i < this.bias.length; i++) {
            let j = this.bias.length - i;
            weighted_errors.push(Matrix.multiply(Matrix.transpose(this.weights[j]), weighted_errors[i - 1]));
        }

        // Calculate the gradients for descent
        let gradients = [];
        for (let i = 1; i < this.forw_prop.length; i++) {
            let j = this.forw_prop.length - i - 1;
            gradients.push(Matrix.map(this.forw_prop[i], this.dactivation));
            gradients[i - 1].multiply(weighted_errors[j]);
            gradients[i - 1].multiply(this.learning_rate);
        }

        // Calculate the differences
        let FP_T = []
        for (let i = 0; i < this.forw_prop.length; i++) {
            FP_T.push(Matrix.transpose(this.forw_prop[i]))
        }
        let weights_diff = []
        for (let i = 0; i < gradients.length; i++) {
            weights_diff.push(Matrix.multiply(gradients[i], FP_T[i]));
        }
        let bias_diff = gradients;

        // Changing weights and bias
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] = Matrix.add(this.weights[i], weights_diff[i])
        }
        for (let i = 0; i < this.bias.length; i++) {
            this.bias[i] = Matrix.add(this.bias[i], bias_diff[i])
        }
    }

    mutate() {
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i].map(this.randomise)
        }
        for (let i = 0; i < this.bias.length; i++) {
            this.bias[i].map(this.randomise)
        }
    }

    static load_trained_NN(loadNN) {
        let newNN = new NN(loadNN.no_inp_nodes, loadNN.no_ans_nodes, loadNN.no_hidden_layers, loadNN.no_hidden_nodes);
        newNN.initialise();
        for (let i = 0; i < loadNN.bias.length; i++) {
            newNN.bias[i] = Matrix.fromArray(Matrix.toArray(loadNN.bias[i]));
        }
        for (let i = 0; i < loadNN.weights.length; i++) {
            newNN.weights[i] = Matrix.fromArray(Matrix.toArray(loadNN.weights[i]));
        }
        return newNN;
    }

    static sigmoid(number) {
        let solution = number;
        solution = (1 / (1 + Math.pow(2.71828, -number)));
        return solution;
    }

    static dsigmoid(number) {
        let solution;
        solution = number * (1 - number);
        return solution;
    }

    static linear(number) {
        return number;
    }

    static dlinear(number) {
        return 1;
    }

    static tanh(number) {
        let solution = (2 / (1 + Math.pow(2.71828, -2 * number))) - 1;
        return solution;
    }

    static dtanh(number) {
        let solution = 1 - Math.pow(tanh(number), 2);
    }

    static relu(number) {
        let solution = number;
        if (solution < 0) {
            solution = 0;
        }
        return solution;
    }

    static drelu(number) {
        let solution = 1;
        if (number < 0) {
            solution = 0;
        }
        return solution;
    }

    static elu(number) {
        let solution = number;
        let alpha = 1;
        if (solution < 0) {
            solution = alpha * (Math.pow(2.71828, number) - 1);
        }
        return solution;
    }

    static delu(number) {
        let solution = 1;
        let alpha = 1;
        if (number < 0) {
            solution = elu(number) + alpha;
        }
        return solution;
    }

    static randomise(number) {
        let solution = number;
        let rate = 0.1;
        if (Math.random() < rate) {
            solution = solution + (Math.random() * 2 - 1) * 0.1
        }
        return solution;
    }

    randomise(number) {
        let solution = number;
        let rate = 0.1;
        if (Math.random() < rate) {
            solution = solution + (Math.random() * 2 - 1) * 0.1
        }
        return solution;
    }
}
