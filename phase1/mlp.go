// Package phase1 implements Multi-Layer Perceptron (MLP) neural network
// Mathematical Foundation: Universal approximation theorem and backpropagation
// Learning Goal: Understanding deep neural networks and gradient-based learning
package phase1

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"
)

// ActivationFunction represents different activation functions
// Mathematical Foundation: Non-linear transformations for universal approximation
type ActivationFunction int

const (
	// Sigmoid activation: σ(x) = 1 / (1 + e^(-x))
	// Range: (0, 1), Smooth gradient
	Sigmoid ActivationFunction = iota

	// Tanh activation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
	// Range: (-1, 1), Zero-centered
	Tanh

	// ReLU activation: max(0, x)
	// Range: [0, +∞), Sparse activation, addresses vanishing gradient
	ReLU
)

// Layer represents a single layer in the MLP
// Mathematical Model: y = f(Wx + b) where f is activation function
type Layer struct {
	Weights    [][]float64        // Weight matrix [neurons][inputs]
	Biases     []float64          // Bias vector [neurons]
	Activation ActivationFunction // Activation function type

	// Cache for backpropagation (stored during forward pass)
	LastInput  []float64 // Input to this layer
	LastOutput []float64 // Output from this layer (after activation)
	LastZ      []float64 // Pre-activation values (before activation)
}

// MLP represents a Multi-Layer Perceptron neural network
// Mathematical Foundation: Composition of linear transformations and non-linearities
// Learning Goal: Understanding how multiple layers enable non-linear pattern recognition
type MLP struct {
	Layers       []*Layer // Network layers (excluding input layer)
	LearningRate float64  // Learning rate (α) for gradient descent
	InputSize    int      // Number of input features
}

// NewMLP creates a new Multi-Layer Perceptron
// Learning Rationale: Flexible architecture allows experimentation with different topologies
// Parameters:
//   - inputSize: number of input features
//   - hiddenSizes: number of neurons in each hidden layer
//   - outputSize: number of output neurons
//   - activations: activation function for each layer (including output)
//   - learningRate: step size for gradient descent
func NewMLP(inputSize int, hiddenSizes []int, outputSize int, activations []ActivationFunction, learningRate float64) (*MLP, error) {
	if len(hiddenSizes)+1 != len(activations) {
		return nil, errors.New("number of activations must equal number of layers (hidden + output)")
	}

	mlp := &MLP{
		Layers:       make([]*Layer, 0, len(hiddenSizes)+1),
		LearningRate: learningRate,
		InputSize:    inputSize,
	}

	// Calculate layer sizes including input
	layerSizes := make([]int, 0, len(hiddenSizes)+2)
	layerSizes = append(layerSizes, inputSize)
	layerSizes = append(layerSizes, hiddenSizes...)
	layerSizes = append(layerSizes, outputSize)

	// Create each layer (excluding input layer)
	for i := 1; i < len(layerSizes); i++ {
		layer := &Layer{
			Weights:    make([][]float64, layerSizes[i]),
			Biases:     make([]float64, layerSizes[i]),
			Activation: activations[i-1],
			LastInput:  make([]float64, layerSizes[i-1]),
			LastOutput: make([]float64, layerSizes[i]),
			LastZ:      make([]float64, layerSizes[i]),
		}

		// Initialize weights using Xavier/Glorot initialization
		// Mathematical Rationale: Maintains activation variance across layers
		variance := 2.0 / float64(layerSizes[i-1]+layerSizes[i])
		stddev := math.Sqrt(variance)

		for j := 0; j < layerSizes[i]; j++ {
			layer.Weights[j] = make([]float64, layerSizes[i-1])
			for k := 0; k < layerSizes[i-1]; k++ {
				// Xavier initialization: weights ~ N(0, 2/(fan_in + fan_out))
				// Note: Using math/rand for deterministic learning experiments
				layer.Weights[j][k] = rand.NormFloat64() * stddev // #nosec G404
			}
			// Initialize biases to zero
			layer.Biases[j] = 0.0
		}

		mlp.Layers = append(mlp.Layers, layer)
	}

	return mlp, nil
}

// activate applies the activation function to input
// Mathematical Foundation: Non-linear transformations enable universal approximation
// Learning Goal: Understanding how different activations affect learning dynamics
func (mlp *MLP) activate(x float64, activation ActivationFunction) float64 {
	switch activation {
	case Sigmoid:
		// Mathematical: σ(x) = 1 / (1 + e^(-x))
		// Prevents overflow for large negative values
		if x < -500 {
			return 0.0
		}
		if x > 500 {
			return 1.0
		}
		return 1.0 / (1.0 + math.Exp(-x))

	case Tanh:
		// Mathematical: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
		// Equivalent to: tanh(x) = 2σ(2x) - 1
		return math.Tanh(x)

	case ReLU:
		// Mathematical: ReLU(x) = max(0, x)
		// Addresses vanishing gradient problem
		return math.Max(0.0, x)

	default:
		return x
	}
}

// activateDerivative computes the derivative of activation function
// Mathematical Foundation: Required for backpropagation gradient computation
// Learning Goal: Understanding how gradients flow through different activations
func (mlp *MLP) activateDerivative(x float64, activation ActivationFunction) float64 {
	switch activation {
	case Sigmoid:
		// Mathematical: σ'(x) = σ(x)(1 - σ(x))
		// Uses output value for efficiency: f'(x) = f(x)(1 - f(x))
		activated := mlp.activate(x, Sigmoid)
		return activated * (1.0 - activated)

	case Tanh:
		// Mathematical: tanh'(x) = 1 - tanh²(x)
		// Uses output value for efficiency: f'(x) = 1 - f(x)²
		activated := math.Tanh(x)
		return 1.0 - activated*activated

	case ReLU:
		// Mathematical: ReLU'(x) = 1 if x > 0, else 0
		// Note: technically undefined at x=0, but we use 0
		if x > 0 {
			return 1.0
		}
		return 0.0

	default:
		return 1.0
	}
}

// Forward performs forward propagation through the network
// Mathematical Foundation: y = f_L(W_L * f_{L-1}(...f_1(W_1 * x + b_1)...) + b_L)
// Learning Goal: Understanding how information flows through neural networks
func (mlp *MLP) Forward(inputs []float64) ([]float64, error) {
	if len(inputs) != mlp.InputSize {
		return nil, errors.New("input size mismatch")
	}

	// Start with input layer
	currentOutput := make([]float64, len(inputs))
	copy(currentOutput, inputs)

	// Propagate through each layer
	for layerIdx, layer := range mlp.Layers {
		// Store input for backpropagation
		copy(layer.LastInput, currentOutput)

		// Calculate pre-activation values: z = W*x + b
		nextOutput := make([]float64, len(layer.Biases))
		for i := 0; i < len(layer.Biases); i++ {
			// Mathematical: z_i = Σ(w_ij * x_j) + b_i
			weightedSum := layer.Biases[i] // Start with bias
			for j := 0; j < len(currentOutput); j++ {
				weightedSum += layer.Weights[i][j] * currentOutput[j]
			}
			layer.LastZ[i] = weightedSum

			// Apply activation function: a = f(z)
			nextOutput[i] = mlp.activate(weightedSum, layer.Activation)
		}

		// Store output for backpropagation
		copy(layer.LastOutput, nextOutput)
		currentOutput = nextOutput

		// Debug information for learning
		_ = layerIdx // Placeholder for potential logging
	}

	return currentOutput, nil
}

// Backward performs backpropagation to compute gradients
// Mathematical Foundation: Chain rule for computing ∂L/∂w and ∂L/∂b
// Learning Goal: Understanding how errors propagate backwards through the network
func (mlp *MLP) Backward(targets []float64) error {
	if len(targets) != len(mlp.Layers[len(mlp.Layers)-1].LastOutput) {
		return errors.New("target size mismatch")
	}

	// Calculate output layer error
	// Mathematical: δ_L = (a_L - y) ⊙ f'(z_L) where ⊙ is element-wise product
	outputLayer := mlp.Layers[len(mlp.Layers)-1]
	outputDeltas := make([]float64, len(outputLayer.LastOutput))

	for i := 0; i < len(outputLayer.LastOutput); i++ {
		// Error signal: difference between prediction and target
		errorSignal := outputLayer.LastOutput[i] - targets[i]

		// Apply activation derivative: δ = error * f'(z)
		derivative := mlp.activateDerivative(outputLayer.LastZ[i], outputLayer.Activation)
		outputDeltas[i] = errorSignal * derivative
	}

	// Propagate errors backwards through hidden layers
	currentDeltas := outputDeltas

	for layerIdx := len(mlp.Layers) - 1; layerIdx >= 0; layerIdx-- {
		layer := mlp.Layers[layerIdx]

		// Update weights: w_ij = w_ij - α * δ_i * a_j
		// Mathematical: ∂L/∂w_ij = δ_i * a_j (from previous layer)
		for i := 0; i < len(layer.Weights); i++ {
			for j := 0; j < len(layer.Weights[i]); j++ {
				gradient := currentDeltas[i] * layer.LastInput[j]
				layer.Weights[i][j] -= mlp.LearningRate * gradient
			}
		}

		// Update biases: b_i = b_i - α * δ_i
		// Mathematical: ∂L/∂b_i = δ_i
		for i := 0; i < len(layer.Biases); i++ {
			layer.Biases[i] -= mlp.LearningRate * currentDeltas[i]
		}

		// Calculate deltas for previous layer (if not first layer)
		if layerIdx > 0 {
			prevLayer := mlp.Layers[layerIdx-1]
			prevDeltas := make([]float64, len(prevLayer.LastOutput))

			for j := 0; j < len(prevDeltas); j++ {
				// Mathematical: δ_j = (Σ w_ij * δ_i) * f'(z_j)
				errorSum := 0.0
				for i := 0; i < len(currentDeltas); i++ {
					errorSum += layer.Weights[i][j] * currentDeltas[i]
				}

				// Apply activation derivative from previous layer
				derivative := mlp.activateDerivative(prevLayer.LastZ[j], prevLayer.Activation)
				prevDeltas[j] = errorSum * derivative
			}

			currentDeltas = prevDeltas
		}
	}

	return nil
}

// Train performs one training iteration
// Learning Goal: Understanding the complete learning cycle
func (mlp *MLP) Train(inputs []float64, targets []float64) error {
	// Step 1: Forward propagation
	_, err := mlp.Forward(inputs)
	if err != nil {
		return err
	}

	// Step 2: Backward propagation
	return mlp.Backward(targets)
}

// TrainBatch trains the network on a batch of examples
// Learning Goal: Understanding batch training vs online learning
func (mlp *MLP) TrainBatch(inputBatch [][]float64, targetBatch [][]float64) error {
	if len(inputBatch) != len(targetBatch) {
		return errors.New("input and target batch sizes must match")
	}

	for i := range inputBatch {
		err := mlp.Train(inputBatch[i], targetBatch[i])
		if err != nil {
			return err
		}
	}

	return nil
}

// Predict performs inference on input data
func (mlp *MLP) Predict(inputs []float64) ([]float64, error) {
	return mlp.Forward(inputs)
}

// CalculateError computes mean squared error
// Mathematical: MSE = (1/n) * Σ(y_i - ŷ_i)²
func (mlp *MLP) CalculateError(inputs [][]float64, targets [][]float64) (float64, error) {
	if len(inputs) != len(targets) {
		return 0, errors.New("input and target batch sizes must match")
	}

	totalError := 0.0
	totalSamples := 0

	for i := range inputs {
		output, err := mlp.Predict(inputs[i])
		if err != nil {
			return 0, err
		}

		// Calculate squared error for this sample
		for j := range output {
			diff := targets[i][j] - output[j]
			totalError += diff * diff
		}
		totalSamples += len(output)
	}

	// Return mean squared error
	return totalError / float64(totalSamples), nil
}

// ToJSON serializes the MLP to JSON for persistence
func (mlp *MLP) ToJSON() ([]byte, error) {
	return json.Marshal(mlp)
}

// FromJSON deserializes MLP from JSON
func (mlp *MLP) FromJSON(data []byte) error {
	return json.Unmarshal(data, mlp)
}
