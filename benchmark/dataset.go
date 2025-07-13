package benchmark

import (
	"fmt"
	"math"
	"math/rand"
)

// Dataset represents a dataset for neural network benchmarking
type Dataset struct {
	Name         string      `json:"name"`
	Description  string      `json:"description"`
	TrainInputs  [][]float64 `json:"train_inputs"`
	TrainTargets [][]float64 `json:"train_targets"`
	TestInputs   [][]float64 `json:"test_inputs"`
	TestTargets  [][]float64 `json:"test_targets"`
	InputSize    int         `json:"input_size"`
	OutputSize   int         `json:"output_size"`
	TrainSize    int         `json:"train_size"`
	TestSize     int         `json:"test_size"`
}

// DatasetBuilder provides a fluent interface for building datasets
type DatasetBuilder struct {
	dataset Dataset
}

// NewDatasetBuilder creates a new dataset builder
func NewDatasetBuilder(name string) *DatasetBuilder {
	return &DatasetBuilder{
		dataset: Dataset{
			Name:         name,
			TrainInputs:  [][]float64{},
			TrainTargets: [][]float64{},
			TestInputs:   [][]float64{},
			TestTargets:  [][]float64{},
		},
	}
}

// WithDescription sets the dataset description
func (db *DatasetBuilder) WithDescription(description string) *DatasetBuilder {
	db.dataset.Description = description
	return db
}

// AddTrainExample adds a training example
func (db *DatasetBuilder) AddTrainExample(input []float64, target []float64) *DatasetBuilder {
	db.dataset.TrainInputs = append(db.dataset.TrainInputs, input)
	db.dataset.TrainTargets = append(db.dataset.TrainTargets, target)
	return db
}

// AddTestExample adds a test example
func (db *DatasetBuilder) AddTestExample(input []float64, target []float64) *DatasetBuilder {
	db.dataset.TestInputs = append(db.dataset.TestInputs, input)
	db.dataset.TestTargets = append(db.dataset.TestTargets, target)
	return db
}

// Build finalizes the dataset construction
func (db *DatasetBuilder) Build() Dataset {
	dataset := db.dataset

	// Set metadata
	if len(dataset.TrainInputs) > 0 {
		dataset.InputSize = len(dataset.TrainInputs[0])
		dataset.TrainSize = len(dataset.TrainInputs)
	}
	if len(dataset.TrainTargets) > 0 {
		dataset.OutputSize = len(dataset.TrainTargets[0])
	}
	if len(dataset.TestInputs) > 0 {
		dataset.TestSize = len(dataset.TestInputs)
	}

	return dataset
}

// CreateXORDataset creates the classic XOR problem dataset
// Mathematical Foundation: Non-linearly separable problem
// XOR truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
func CreateXORDataset() Dataset {
	builder := NewDatasetBuilder("xor").
		WithDescription("XOR logical operation - classic non-linearly separable problem")

	// XOR truth table as training data
	xorData := []struct {
		input  []float64
		output []float64
	}{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	// Add training examples (use all examples for training)
	for _, example := range xorData {
		builder.AddTrainExample(example.input, example.output)
	}

	// Use same data for testing (small dataset)
	for _, example := range xorData {
		builder.AddTestExample(example.input, example.output)
	}

	return builder.Build()
}

// CreateANDDataset creates the AND logical operation dataset
// Mathematical Foundation: Linearly separable problem
func CreateANDDataset() Dataset {
	builder := NewDatasetBuilder("and").
		WithDescription("AND logical operation - linearly separable problem")

	// AND truth table
	andData := []struct {
		input  []float64
		output []float64
	}{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{0}},
		{[]float64{1, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{1}},
	}

	for _, example := range andData {
		builder.AddTrainExample(example.input, example.output)
		builder.AddTestExample(example.input, example.output)
	}

	return builder.Build()
}

// CreateORDataset creates the OR logical operation dataset
// Mathematical Foundation: Linearly separable problem
func CreateORDataset() Dataset {
	builder := NewDatasetBuilder("or").
		WithDescription("OR logical operation - linearly separable problem")

	// OR truth table
	orData := []struct {
		input  []float64
		output []float64
	}{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{1}},
	}

	for _, example := range orData {
		builder.AddTrainExample(example.input, example.output)
		builder.AddTestExample(example.input, example.output)
	}

	return builder.Build()
}

// CreateLinearSeparableDataset creates a linearly separable 2D classification dataset
// Mathematical Foundation: Data points separated by a line
func CreateLinearSeparableDataset(numSamples int, seed int64) Dataset {
	rand.Seed(seed) //nolint:staticcheck // Deterministic datasets for benchmarking

	builder := NewDatasetBuilder("linear_separable").
		WithDescription("Randomly generated linearly separable 2D classification problem")

	// Define a separating line: y = 0.5*x + 0.3
	// Points above the line are class 1, below are class 0
	for i := 0; i < numSamples; i++ {
		x1 := rand.Float64()*2 - 1 //nolint:gosec // Non-crypto random for benchmark data
		x2 := rand.Float64()*2 - 1 //nolint:gosec // Non-crypto random for benchmark data

		// Determine class based on line equation
		separatingValue := 0.5*x1 + 0.3
		var class float64
		if x2 > separatingValue {
			class = 1.0
		} else {
			class = 0.0
		}

		input := []float64{x1, x2}
		target := []float64{class}

		// 80% for training, 20% for testing
		if i < int(float64(numSamples)*0.8) {
			builder.AddTrainExample(input, target)
		} else {
			builder.AddTestExample(input, target)
		}
	}

	return builder.Build()
}

// CreateNonLinearDataset creates a non-linearly separable dataset
// Mathematical Foundation: Circular decision boundary
func CreateNonLinearDataset(numSamples int, seed int64) Dataset {
	rand.Seed(seed) //nolint:staticcheck // Deterministic datasets for benchmarking

	builder := NewDatasetBuilder("non_linear").
		WithDescription("Randomly generated non-linearly separable dataset with circular boundary")

	// Circular decision boundary: x1² + x2² = 0.5
	// Points inside circle are class 1, outside are class 0
	for i := 0; i < numSamples; i++ {
		x1 := rand.Float64()*2 - 1 //nolint:gosec // Non-crypto random for benchmark data
		x2 := rand.Float64()*2 - 1 //nolint:gosec // Non-crypto random for benchmark data

		// Calculate distance from origin
		distanceSquared := x1*x1 + x2*x2
		var class float64
		if distanceSquared < 0.5 {
			class = 1.0
		} else {
			class = 0.0
		}

		input := []float64{x1, x2}
		target := []float64{class}

		// 80% for training, 20% for testing
		if i < int(float64(numSamples)*0.8) {
			builder.AddTrainExample(input, target)
		} else {
			builder.AddTestExample(input, target)
		}
	}

	return builder.Build()
}

// CreateSinusoidalDataset creates a sinusoidal regression dataset
// Mathematical Foundation: y = sin(x) function approximation
func CreateSinusoidalDataset(numSamples int, noiseLevel float64, seed int64) Dataset {
	rand.Seed(seed) //nolint:staticcheck // Deterministic datasets for benchmarking

	builder := NewDatasetBuilder("sinusoidal").
		WithDescription("Sinusoidal function approximation with optional noise")

	for i := 0; i < numSamples; i++ {
		// Generate x values from 0 to 2π
		x := float64(i) / float64(numSamples) * 2 * math.Pi

		// Calculate y = sin(x) with optional noise
		y := math.Sin(x)
		if noiseLevel > 0 {
			noise := (rand.Float64()*2 - 1) * noiseLevel //nolint:gosec // Non-crypto random for benchmark data
			y += noise
		}

		input := []float64{x}
		target := []float64{y}

		// 80% for training, 20% for testing
		if i < int(float64(numSamples)*0.8) {
			builder.AddTrainExample(input, target)
		} else {
			builder.AddTestExample(input, target)
		}
	}

	return builder.Build()
}

// GetStandardDatasets returns all standard benchmark datasets
func GetStandardDatasets() []Dataset {
	return []Dataset{
		CreateXORDataset(),
		CreateANDDataset(),
		CreateORDataset(),
		CreateLinearSeparableDataset(100, 42),
		CreateNonLinearDataset(200, 42),
	}
}

// ValidateDataset performs validation checks on a dataset
func ValidateDataset(dataset Dataset) error {
	// Check if dataset has examples
	if len(dataset.TrainInputs) == 0 {
		return fmt.Errorf("dataset '%s' has no training examples", dataset.Name)
	}

	if len(dataset.TestInputs) == 0 {
		return fmt.Errorf("dataset '%s' has no test examples", dataset.Name)
	}

	// Check input/target consistency
	if len(dataset.TrainInputs) != len(dataset.TrainTargets) {
		return fmt.Errorf("dataset '%s' has mismatched training inputs and targets", dataset.Name)
	}

	if len(dataset.TestInputs) != len(dataset.TestTargets) {
		return fmt.Errorf("dataset '%s' has mismatched test inputs and targets", dataset.Name)
	}

	// Check input size consistency
	inputSize := len(dataset.TrainInputs[0])
	for i, input := range dataset.TrainInputs {
		if len(input) != inputSize {
			return fmt.Errorf("dataset '%s' has inconsistent input size at training example %d", dataset.Name, i)
		}
	}

	for i, input := range dataset.TestInputs {
		if len(input) != inputSize {
			return fmt.Errorf("dataset '%s' has inconsistent input size at test example %d", dataset.Name, i)
		}
	}

	// Check target size consistency
	targetSize := len(dataset.TrainTargets[0])
	for i, target := range dataset.TrainTargets {
		if len(target) != targetSize {
			return fmt.Errorf("dataset '%s' has inconsistent target size at training example %d", dataset.Name, i)
		}
	}

	for i, target := range dataset.TestTargets {
		if len(target) != targetSize {
			return fmt.Errorf("dataset '%s' has inconsistent target size at test example %d", dataset.Name, i)
		}
	}

	return nil
}

// PrintDatasetInfo displays dataset statistics
func PrintDatasetInfo(dataset Dataset) {
	fmt.Printf("📊 Dataset: %s\n", dataset.Name)
	fmt.Printf("   Description: %s\n", dataset.Description)
	fmt.Printf("   Input size: %d\n", dataset.InputSize)
	fmt.Printf("   Output size: %d\n", dataset.OutputSize)
	fmt.Printf("   Training examples: %d\n", dataset.TrainSize)
	fmt.Printf("   Test examples: %d\n", dataset.TestSize)

	// Show sample examples
	if len(dataset.TrainInputs) > 0 {
		fmt.Printf("   Sample input: %v\n", dataset.TrainInputs[0])
		fmt.Printf("   Sample target: %v\n", dataset.TrainTargets[0])
	}
}
