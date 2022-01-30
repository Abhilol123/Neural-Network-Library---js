class NN {
	constructor(layer_info, activation = 'sigmoid', learning_rate = 0.1) {
		// Expecting only a layer_info as an array of integers.
		if (!layer_info) {
			console.error("ERROR: layer_info is undefined");
			return;
		}
		for (let i = 0; i < layer_info.length; i++) {
			if (!parseInt(layer_info[i])) {
				console.error("ERROR: layer_info has bad or null value");
				return;
			};
		}
		this.layer_info = layer_info;
		this.forw_prop;
		this.activation = activation;
		this.learning_rate = learning_rate;
		this.learning_dataset;
		this.activation_function = {
			'sigmoid': function (number) { return (1 / (1 + Math.pow(2.71828, -number))); },
			'linear': function (number) { return (number); },
			'tanh': function (number) { return ((2 / (1 + Math.pow(2.71828, -2 * number))) - 1); },
			'relu': function (number) { return (number < 0 ? 0 : number); },
			'elu': function (number, alpha = 1) { return (number < 0 ? (alpha * (Math.pow(2.71828, number) - 1)) : number); }
		};
		this.deactivation_function = {
			'sigmoid': function (number) { return (number * (1 - number)); },
			'linear': function (number) { return 1; },
			'tanh': function (number) { return (1 - Math.pow(tanh(number), 2)); },
			'relu': function (number) { return (number < 0 ? 0 : 1); },
			'elu': function (number, alpha = 1) { return (number < 0 ? ((number < 0 ? (alpha * (Math.pow(2.71828, number) - 1)) : number) + alpha) : 1); }
		};
		this.weights = [];
		this.bias = [];
		for (let i = 0; i < this.layer_info.length - 1; i++) {
			this.weights.push(new Matrix(this.layer_info[i + 1], this.layer_info[i]));
			this.bias.push(new Matrix(this.layer_info[i + 1], 1));
		}
	}

	forwardPropagation(inputs_matrix) {
		// this function is for using the neural netwrk to give outputs
		// It expects a Matrix of certain shape.
		if (!inputs_matrix || !inputs_matrix.rows || !inputs_matrix.cols || inputs_matrix.rows !== this.layer_info[0] || inputs_matrix.cols !== 1) {
			console.error("ERROR: NO_OR_BAD_value_of_inputs_matrix: ", inputs_matrix);
			return;
		}
		this.forw_prop = [inputs_matrix];
		for (let i = 0; i < this.bias.length; i++) {
			this.forw_prop.push(Matrix.multiply(this.weights[i], this.forw_prop[i]));
			this.forw_prop[i + 1] = Matrix.add(this.forw_prop[i + 1], this.bias[i]);
			this.forw_prop[i + 1].map(this.activation_function[this.activation]);
		}
		return this.forw_prop[this.forw_prop.length - 1];
	}

	backPropagation(expected_output_matrix, learning_rate) {
		// This function is to train the neural network.
		// It expects a Matrix of certain shape.
		if (!expected_output_matrix || !expected_output_matrix.rows || !expected_output_matrix.cols || expected_output_matrix.rows !== this.layer_info[this.layer_info.length - 1] || expected_output_matrix.cols !== 1) {
			console.error("ERROR: NO_OR_BAD_value_of_expected_output_matrix: ", expected_output_matrix);
			return;
		}
		if (learning_rate) {
			this.learning_rate = learning_rate;
		}

		// Calculate the weighted errors
		let weighted_errors = [Matrix.subtract(expected_output_matrix, this.forw_prop[this.forw_prop.length - 1])];
		for (let i = 1; i < this.bias.length; i++) {
			weighted_errors.push(Matrix.multiply(Matrix.transpose(this.weights[this.bias.length - i]), weighted_errors[i - 1]));
		}

		// Calculate the gradients for descent
		let gradients = [];
		for (let i = 1; i < this.forw_prop.length; i++) {
			gradients.push(Matrix.map(this.forw_prop[i], this.deactivation_function[this.activation]));
			gradients[i - 1].multiply(weighted_errors[this.forw_prop.length - i - 1]);
			gradients[i - 1].multiply(this.learning_rate);
		}

		// Calculate the differences
		let bias_diff = gradients;
		let weights_diff = [];
		let forwardPropTranspose = [];
		for (let i = 0; i < this.forw_prop.length; i++) {
			forwardPropTranspose.push(Matrix.transpose(this.forw_prop[i]));
		}
		for (let i = 0; i < gradients.length; i++) {
			weights_diff.push(Matrix.multiply(gradients[i], forwardPropTranspose[i]));
		}

		// Changing weights and bias
		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i] = Matrix.add(this.weights[i], weights_diff[i]);
		}
		for (let i = 0; i < this.bias.length; i++) {
			this.bias[i] = Matrix.add(this.bias[i], bias_diff[i]);
		}
	}

	learn(input_output_pair, learning_rate) {
		// This function expects the data to be in the form of an 'array of object' with keys 'input' and 'output'
		// The input and output are expected to be instances of Matrix
		if (learning_rate) {
			this.learning_rate = learning_rate;
		}
		if (input_output_pair) {
			this.learning_dataset = input_output_pair;
		}
		for (let i = 0; i < this.learning_dataset.length; i++) {
			let index = parseInt(this.learning_dataset.length * Math.random());
			index = index >= this.learning_dataset.length ? index - 1 : index;
			this.forwardPropagation(this.learning_dataset[index].input);
			this.backPropagation(this.learning_dataset[index].output);
		}
	}

	mutate() {
		// This function is to randomly generate some variation for the weights and bias if learning becomes stagnent
		let randomise = (number, rate = 0.1) => {
			if (Math.random() < rate) {
				let temp = number + (Math.random() * 2 - 1) * rate;
				return (temp > 1) ? 1 : ((temp < -1) ? -1 : temp);
			} else {
				return number;
			}
		}
		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i].map(randomise);
		}
		for (let i = 0; i < this.bias.length; i++) {
			this.bias[i].map(randomise);
		}
	}

	static load_trained_NN(nn_object) {
		// This is called to load a neural network modal.
		function convertObject(inputVariable) {
			if (inputVariable) {
				switch (Object.prototype.toString.call(inputVariable)) {
					case '[object Array]':
						if (inputVariable[0] && inputVariable[0].rows && inputVariable[0].cols && inputVariable[0].data) {
							let newArr = []
							for (let i = 0; i < inputVariable.length; i++) {
								newArr.push(Matrix.fromArray(inputVariable[i].data))
							}
							return newArr;
						}
						if (inputVariable[0] && Object.prototype.toString.call(inputVariable[0]) === '[object Number]') {
							return inputVariable;
						}
						return null;
					case '[object String]':
					case '[object Number]':
						return inputVariable;
					case '[object Object]':
					default:
						return null;
				}
			} else {
				return null;
			}
		}
		let newNN = new NN(nn_object.layer_info);
		for (let i = 0; i < Object.keys(nn_object).length; i++) {
			let convertedObject = convertObject(nn_object[Object.keys(nn_object)[i]]);
			if (convertedObject) {
				newNN[[Object.keys(nn_object)[i]]] = convertedObject;
			}
		}
		return newNN;
	}

	static save_trained_NN(nn_object) {
		// This function is called to save a NN
		return JSON.stringify(nn_object);
	}
}

if (typeof module !== 'undefined') {
	module.exports = NN;
}
