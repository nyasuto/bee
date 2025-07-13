// Package model defines interfaces for model management operations
package model

// Model represents a trainable machine learning model
type Model interface {
	// Train trains the model with the given data
	Train(inputs [][]float64, targets []float64, epochs int) (actualEpochs int, err error)

	// Predict makes a prediction for the given input
	Predict(input []float64) (float64, error)

	// Accuracy calculates the accuracy on the given dataset
	Accuracy(inputs [][]float64, targets []float64) (float64, error)

	// GetWeights returns the model weights (for inspection)
	GetWeights() []float64

	// GetBias returns the model bias (for inspection)
	GetBias() float64
}

// ModelManager handles model creation, persistence, and lifecycle
type ModelManager interface {
	// CreateModel creates a new model of the specified type
	CreateModel(modelType string, inputSize int, learningRate float64) (Model, error)

	// SaveModel saves a model to the specified path
	SaveModel(model Model, path string) error

	// LoadModel loads a model from the specified path
	LoadModel(path string) (Model, error)

	// ListModelTypes returns available model types
	ListModelTypes() []string
}

// ModelConfig represents configuration for model creation
type ModelConfig struct {
	Type         string
	InputSize    int
	LearningRate float64
	HiddenLayers []int // For MLP models
}
