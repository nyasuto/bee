// Package phase1 implements basic perceptron neural network
// Mathematical Foundation: McCulloch-Pitts neuron model (1943)
// Learning Goal: Understanding linear classification and weight updates
package phase1

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Perceptron represents a basic perceptron neuron
// Mathematical Model: y = σ(w·x + b) where σ is activation function
type Perceptron struct {
	Weights      []float64 `json:"weights"`       // synaptic weights (w)
	Bias         float64   `json:"bias"`          // bias term (b)
	LearningRate float64   `json:"learning_rate"` // learning rate (α)
	Epochs       int       `json:"epochs"`        // training epochs
}

// NewPerceptron creates a new perceptron with random weights
// Learning Rationale: Understanding initialization strategies
func NewPerceptron(inputSize int, learningRate float64) *Perceptron {
	// Initialize with current time for random initialization
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	weights := make([]float64, inputSize)
	// Xavier initialization for better convergence
	// Mathematical: w_i ~ U(-√(6/n), √(6/n)) where n is input size
	limit := math.Sqrt(6.0 / float64(inputSize))
	for i := range weights {
		weights[i] = (rng.Float64()*2 - 1) * limit
	}

	return &Perceptron{
		Weights:      weights,
		Bias:         0.0, // Start with zero bias
		LearningRate: learningRate,
		Epochs:       1000, // Default epochs
	}
}

// NewPerceptronWithSeed creates a perceptron with specific seed for reproducible testing
func NewPerceptronWithSeed(inputSize int, learningRate float64, seed int64) *Perceptron {
	// Create a new random source with the given seed
	rng := rand.New(rand.NewSource(seed))

	weights := make([]float64, inputSize)
	limit := math.Sqrt(6.0 / float64(inputSize))
	for i := range weights {
		weights[i] = (rng.Float64()*2 - 1) * limit
	}

	return &Perceptron{
		Weights:      weights,
		Bias:         0.0,
		LearningRate: learningRate,
		Epochs:       1000,
	}
}

// Forward performs forward propagation
// Mathematical Foundation: y = σ(Σ(wi * xi) + b)
// Learning Goal: Understanding weighted sum and activation
func (p *Perceptron) Forward(inputs []float64) (float64, error) {
	if len(inputs) != len(p.Weights) {
		return 0, fmt.Errorf("input size mismatch: expected %d, got %d",
			len(p.Weights), len(inputs))
	}

	// Step 1: Calculate weighted sum (明示的実装)
	// Mathematical: z = Σ(wi * xi) + b
	weightedSum := p.Bias
	for i, input := range inputs {
		weightedSum += p.Weights[i] * input
	}

	// Step 2: Apply activation function (Heaviside step function)
	// Mathematical: σ(x) = 1 if x ≥ 0, else 0
	return p.stepFunction(weightedSum), nil
}

// stepFunction implements the Heaviside step function
// Mathematical: σ(x) = 1 if x ≥ 0, else 0
// Learning Goal: Understanding discrete activation functions
func (p *Perceptron) stepFunction(x float64) float64 {
	if x >= 0.0 {
		return 1.0
	}
	return 0.0
}

// Train performs one training iteration using perceptron learning rule
// Mathematical Foundation: Δw = α(t - y)x where t=target, y=output
// Learning Goal: Understanding gradient-free weight updates
func (p *Perceptron) Train(inputs []float64, target float64) error {
	if len(inputs) != len(p.Weights) {
		return fmt.Errorf("input size mismatch: expected %d, got %d",
			len(p.Weights), len(inputs))
	}

	// Step 1: Forward propagation
	output, err := p.Forward(inputs)
	if err != nil {
		return err
	}

	// Step 2: Calculate error
	// Mathematical: e = t - y
	error := target - output

	// Step 3: Update weights only if there's an error (perceptron learning rule)
	// Mathematical: wi = wi + α * error * xi
	if error != 0 {
		for i, input := range inputs {
			p.Weights[i] += p.LearningRate * error * input
		}

		// Step 4: Update bias
		// Mathematical: b = b + α * error
		p.Bias += p.LearningRate * error
	}

	return nil
}

// TrainDataset trains the perceptron on a complete dataset
// Learning Goal: Understanding epoch-based training and convergence
func (p *Perceptron) TrainDataset(inputs [][]float64, targets []float64, maxEpochs int) (int, error) {
	if len(inputs) != len(targets) {
		return 0, errors.New("inputs and targets size mismatch")
	}

	if len(inputs) == 0 {
		return 0, errors.New("empty dataset")
	}

	// Validate input dimensions
	expectedSize := len(p.Weights)
	for i, input := range inputs {
		if len(input) != expectedSize {
			return 0, fmt.Errorf("input[%d] size mismatch: expected %d, got %d",
				i, expectedSize, len(input))
		}
	}

	// Training loop with early stopping on convergence
	for epoch := 0; epoch < maxEpochs; epoch++ {
		totalErrors := 0

		// Train on each sample
		for i, input := range inputs {
			// Get prediction before training
			prediction, err := p.Forward(input)
			if err != nil {
				return epoch, err
			}

			// Count errors before update
			if prediction != targets[i] {
				totalErrors++
			}

			// Update weights
			err = p.Train(input, targets[i])
			if err != nil {
				return epoch, err
			}
		}

		// Early stopping if no errors (perfect classification)
		if totalErrors == 0 {
			return epoch + 1, nil
		}
	}

	return maxEpochs, nil
}

// Predict performs inference on a single input
// Learning Goal: Understanding prediction vs training distinction
func (p *Perceptron) Predict(inputs []float64) (float64, error) {
	return p.Forward(inputs)
}

// Accuracy calculates accuracy on a dataset
// Mathematical: accuracy = correct_predictions / total_predictions
// Learning Goal: Understanding model evaluation metrics
func (p *Perceptron) Accuracy(inputs [][]float64, targets []float64) (float64, error) {
	if len(inputs) != len(targets) {
		return 0, errors.New("inputs and targets size mismatch")
	}

	if len(inputs) == 0 {
		return 0, errors.New("empty dataset")
	}

	correct := 0
	for i, input := range inputs {
		prediction, err := p.Predict(input)
		if err != nil {
			return 0, err
		}

		if prediction == targets[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(inputs)), nil
}

// GetWeights returns a copy of current weights
// Learning Goal: Understanding parameter inspection
func (p *Perceptron) GetWeights() []float64 {
	weights := make([]float64, len(p.Weights))
	copy(weights, p.Weights)
	return weights
}

// GetBias returns current bias value
func (p *Perceptron) GetBias() float64 {
	return p.Bias
}

// ToJSON serializes the perceptron to JSON for model persistence
// Learning Goal: Understanding model save/load patterns
func (p *Perceptron) ToJSON() ([]byte, error) {
	return json.Marshal(p)
}

// FromJSON deserializes a perceptron from JSON
func FromJSON(data []byte) (*Perceptron, error) {
	var p Perceptron
	err := json.Unmarshal(data, &p)
	if err != nil {
		return nil, err
	}
	return &p, nil
}

// String provides a human-readable representation
// Learning Goal: Understanding model introspection
func (p *Perceptron) String() string {
	return fmt.Sprintf("Perceptron{weights: %v, bias: %.4f, lr: %.4f, epochs: %d}",
		p.Weights, p.Bias, p.LearningRate, p.Epochs)
}
