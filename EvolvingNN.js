const Matrix = require("./Matrix.js");

class EvolvingNN {
	constructor(layer_info, activation = 'sigmoid', learning_rate = 0.1, generation = 0) {
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
		this.learning_rate = learning_rate;
		this.weights = [];
		this.bias = [];
		for (let i = 0; i < this.layer_info.length - 1; i++) {
			this.weights.push(new Matrix(this.layer_info[i + 1], this.layer_info[i]));
			this.bias.push(new Matrix(this.layer_info[i + 1], 1));
		}
		this.scrore = 0;
		this.fitness = 0;
		this.generation = generation;
		this.activation = activation;
		this.activation_function = {
			'sigmoid': function (number) { return (1 / (1 + Math.pow(2.71828, -number))); },
			'linear': function (number) { return (number); },
			'tanh': function (number) { return ((2 / (1 + Math.pow(2.71828, -2 * number))) - 1); },
			'relu': function (number) { return (number < 0 ? 0 : number); },
			'elu': function (number, alpha = 1) { return (number < 0 ? (alpha * (Math.pow(2.71828, number) - 1)) : number); }
		};
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

	mutate() {
		// This function is to randomly generate some variation for the weights and bias if learning becomes stagnent
		let randomise = (number) => {
			let temp = number + (Math.random() * 2 - 1) * this.learning_rate;
			return (temp > 1) ? 1 : ((temp < -1) ? -1 : temp);
		}
		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i].map(randomise);
		}
		for (let i = 0; i < this.bias.length; i++) {
			this.bias[i].map(randomise);
		}
	}

	static getNewGen(population, layer_info, activation = 'sigmoid', learning_rate = 0.1) {
		// use this function to get a generation of NN instances
		let newGen = []
		for (let i = 0; i < population; i++) {
			newGen.push(new EvolvingNN(layer_info, activation, learning_rate));
		}
		return newGen
	}

	static getNextGen(oldGen, learning_rate) {
		// Use this function to create a new generation from old gen
		if (learning_rate) {
			this.learning_rate = learning_rate;
		}

		// Calculates the fitness for all old generation NN
		let sum = 0;
		for (let i = 0; i < oldGen.length; i++) {
			sum = sum + oldGen[i].score;
		}
		for (let i = 0; i < oldGen.length; i++) {
			oldGen[i].fitness = oldGen[i].score / sum;
		}
		oldGen.sort(function (a, b) {
			return b.fitness - a.fitness
		});

		// Selecting some from old to move to next round
		let newGen = [];
		let population = oldGen.length;
		for (let i = 0; i < population * 0.05; i++) {
			newGen.push(EvolvingNN.generateCopy(oldGen[i]));
		}
		for (let i = population * 0.05; i < population * 0.80; i++) {
			let index = parseInt(Math.random() * population * 0.05);
			let temp = EvolvingNN.generateCopy(oldGen[index]);
			temp.mutate();
			newGen.push(temp);
		}
		for (let i = population * 0.80; i < population; i++) {
			newGen.push(new EvolvingNN(oldGen[0].layer_info, oldGen[0].activation, oldGen[0].learning_rate, oldGen[0].generation + 1));
		}
		return { newGen: newGen, oldGen: oldGen };
	}

	static generateCopy(oldNN) {
		let newNN = new EvolvingNN(oldNN.layer_info, oldNN.activation, oldNN.learning_rate, oldNN.generation + 1);
		newNN.weights = [];
		newNN.bias = [];
		newNN.forw_prop = [];
		for (let i = 0; i < oldNN.weights.length; i++) {
			newNN.weights.push(Matrix.fromArray(oldNN.weights[i].data));
		}
		for (let i = 0; i < oldNN.bias.length; i++) {
			newNN.bias.push(Matrix.fromArray(oldNN.bias[i].data));
		}
		for (let i = 0; i < oldNN.forw_prop.length; i++) {
			newNN.forw_prop.push(Matrix.fromArray(oldNN.forw_prop[i].data));
		}
		return newNN;
	}

	static load_generation(nn_object, population = 1000) {
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
		let newNN = new EvolvingNN(nn_object.layer_info);
		for (let i = 0; i < Object.keys(nn_object).length; i++) {
			let convertedObject = convertObject(nn_object[Object.keys(nn_object)[i]]);
			if (convertedObject) {
				newNN[[Object.keys(nn_object)[i]]] = convertedObject;
			}
		}
		return newNN;
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
		let newNN = new EvolvingNN(nn_object.layer_info);
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
	module.exports = EvolvingNN;
}
