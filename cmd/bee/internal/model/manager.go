// Package model provides model management implementations
package model

import (
	"fmt"
	"os"
	"strings"

	"github.com/nyasuto/bee/phase1"
)

// DefaultModelManager implements ModelManager
type DefaultModelManager struct{}

// NewDefaultModelManager creates a new default model manager
func NewDefaultModelManager() *DefaultModelManager {
	return &DefaultModelManager{}
}

// CreateModel creates a new model of the specified type
func (m *DefaultModelManager) CreateModel(modelType string, inputSize int, learningRate float64) (Model, error) {
	switch strings.ToLower(modelType) {
	case "perceptron":
		perceptron := phase1.NewPerceptron(inputSize, learningRate)
		return &PerceptronAdapter{perceptron: perceptron}, nil
	default:
		return nil, fmt.Errorf("unsupported model type: %s", modelType)
	}
}

// SaveModel saves a model to the specified path
func (m *DefaultModelManager) SaveModel(model Model, path string) error {
	// Create directory if it doesn't exist
	lastSlash := strings.LastIndex(path, "/")
	if lastSlash != -1 {
		dir := path[:lastSlash]
		if dir != "" {
			err := os.MkdirAll(dir, 0750)
			if err != nil {
				return fmt.Errorf("failed to create directory: %w", err)
			}
		}
	}

	// For now, we only support perceptron models
	if adapter, ok := model.(*PerceptronAdapter); ok {
		data, err := adapter.perceptron.ToJSON()
		if err != nil {
			return fmt.Errorf("failed to serialize model: %w", err)
		}

		err = os.WriteFile(path, data, 0600)
		if err != nil {
			return fmt.Errorf("failed to write file: %w", err)
		}

		return nil
	}

	return fmt.Errorf("unsupported model type for saving")
}

// LoadModel loads a model from the specified path
func (m *DefaultModelManager) LoadModel(path string) (Model, error) {
	// Validate file path to prevent directory traversal
	if strings.Contains(path, "..") || strings.HasPrefix(path, "/") {
		return nil, fmt.Errorf("invalid file path: absolute paths and directory traversal not allowed")
	}

	// #nosec G304 - path is validated above
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	perceptron, err := phase1.FromJSON(data)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize model: %w", err)
	}

	return &PerceptronAdapter{perceptron: perceptron}, nil
}

// ListModelTypes returns available model types
func (m *DefaultModelManager) ListModelTypes() []string {
	return []string{"perceptron"}
}

// PerceptronAdapter adapts phase1.Perceptron to the Model interface
type PerceptronAdapter struct {
	perceptron *phase1.Perceptron
}

// Train trains the model with the given data
func (p *PerceptronAdapter) Train(inputs [][]float64, targets []float64, epochs int) (int, error) {
	return p.perceptron.TrainDataset(inputs, targets, epochs)
}

// Predict makes a prediction for the given input
func (p *PerceptronAdapter) Predict(input []float64) (float64, error) {
	return p.perceptron.Predict(input)
}

// Accuracy calculates the accuracy on the given dataset
func (p *PerceptronAdapter) Accuracy(inputs [][]float64, targets []float64) (float64, error) {
	return p.perceptron.Accuracy(inputs, targets)
}

// GetWeights returns the model weights
func (p *PerceptronAdapter) GetWeights() []float64 {
	return p.perceptron.GetWeights()
}

// GetBias returns the model bias
func (p *PerceptronAdapter) GetBias() float64 {
	return p.perceptron.GetBias()
}
