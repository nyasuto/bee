// Package data defines interfaces for data access operations
package data

// DataLoader defines the interface for loading training data
type DataLoader interface {
	// LoadTrainingData loads training data from the specified path
	LoadTrainingData(path string) (inputs [][]float64, targets []float64, err error)

	// ValidatePath checks if the data path is valid and accessible
	ValidatePath(path string) error
}

// DataValidator validates loaded data
type DataValidator interface {
	// ValidateTrainingData checks if the training data is valid
	ValidateTrainingData(inputs [][]float64, targets []float64) error

	// ValidateInputData validates input data for inference
	ValidateInputData(inputs []float64) error
}

// DataParser parses input strings into numerical data
type DataParser interface {
	// ParseInputString converts comma-separated string to float64 slice
	ParseInputString(data string) ([]float64, error)
}
