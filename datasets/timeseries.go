// Package datasets implements time series dataset generation and evaluation for RNN/LSTM
// Learning Goal: Understanding sequence learning and time series prediction with neural networks
package datasets

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// TimeSeriesDataset represents a time series dataset for sequence learning
// Mathematical Foundation: Sequential data with temporal dependencies
type TimeSeriesDataset struct {
	Name        string        // Dataset name (e.g., "SineWave", "Fibonacci")
	Type        string        // Dataset type ("regression", "classification", "prediction")
	Sequences   [][][]float64 // Sequences: [samples][timesteps][features]
	Targets     [][]float64   // Target values: [samples][outputs]
	Features    int           // Number of input features per timestep
	OutputSize  int           // Number of output values
	SeqLength   int           // Sequence length (fixed for this implementation)
	Description string        // Dataset description
}

// SineWaveConfig configuration for sine wave generation
// Learning Goal: Understanding periodic pattern learning in RNNs
type SineWaveConfig struct {
	NumSamples   int     // Number of sequence samples
	SeqLength    int     // Length of each sequence
	Frequency    float64 // Sine wave frequency
	Amplitude    float64 // Sine wave amplitude
	Phase        float64 // Phase offset
	NoiseLevel   float64 // Gaussian noise level
	SampleRate   float64 // Sampling rate
	PredictSteps int     // Number of future steps to predict
}

// FibonacciConfig configuration for Fibonacci sequence generation
// Learning Goal: Understanding arithmetic sequence learning and pattern recognition
type FibonacciConfig struct {
	NumSamples   int  // Number of sequence samples
	SeqLength    int  // Length of each sequence
	MaxValue     int  // Maximum Fibonacci value
	PredictSteps int  // Number of future steps to predict
	Normalize    bool // Whether to normalize values
}

// RandomWalkConfig configuration for random walk generation
// Learning Goal: Understanding stochastic process modeling
type RandomWalkConfig struct {
	NumSamples int     // Number of sequence samples
	SeqLength  int     // Length of each sequence
	StepSize   float64 // Standard deviation of random steps
	Drift      float64 // Drift coefficient
	StartValue float64 // Starting value
}

// GenerateSineWave creates a sine wave time series dataset
// Mathematical Foundation: y = A * sin(2π * f * t + φ) + noise
// Learning Goal: Understanding periodic pattern learning and prediction
func GenerateSineWave(config SineWaveConfig) (*TimeSeriesDataset, error) {
	if config.NumSamples <= 0 || config.SeqLength <= 0 {
		return nil, errors.New("invalid configuration: samples and sequence length must be positive")
	}
	if config.PredictSteps <= 0 {
		config.PredictSteps = 1 // Default to single-step prediction
	}

	sequences := make([][][]float64, config.NumSamples)
	targets := make([][]float64, config.NumSamples)

	// Generate sequences with different starting points
	for i := 0; i < config.NumSamples; i++ {
		// Random starting point to ensure diversity
		startTime := rand.Float64() * 10.0 //nolint:gosec // Educational implementation

		sequence := make([][]float64, config.SeqLength)
		target := make([]float64, config.PredictSteps)

		// Generate input sequence
		for t := 0; t < config.SeqLength; t++ {
			timeStep := startTime + float64(t)/config.SampleRate
			// Sine wave: A * sin(2π * f * t + φ) + noise
			value := config.Amplitude * math.Sin(2*math.Pi*config.Frequency*timeStep+config.Phase)

			// Add Gaussian noise
			if config.NoiseLevel > 0 {
				noise := rand.NormFloat64() * config.NoiseLevel //nolint:gosec // Educational implementation
				value += noise
			}

			sequence[t] = []float64{value}
		}

		// Generate target sequence (future predictions)
		for p := 0; p < config.PredictSteps; p++ {
			timeStep := startTime + float64(config.SeqLength+p)/config.SampleRate
			targetValue := config.Amplitude * math.Sin(2*math.Pi*config.Frequency*timeStep+config.Phase)
			target[p] = targetValue
		}

		sequences[i] = sequence
		targets[i] = target
	}

	return &TimeSeriesDataset{
		Name:        "SineWave",
		Type:        "regression",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  config.PredictSteps,
		SeqLength:   config.SeqLength,
		Description: fmt.Sprintf("Sine wave dataset: freq=%.2f, amp=%.2f, noise=%.3f", config.Frequency, config.Amplitude, config.NoiseLevel),
	}, nil
}

// GenerateFibonacci creates a Fibonacci sequence dataset
// Mathematical Foundation: F(n) = F(n-1) + F(n-2), F(0)=0, F(1)=1
// Learning Goal: Understanding arithmetic sequence learning and memory requirements
func GenerateFibonacci(config FibonacciConfig) (*TimeSeriesDataset, error) {
	if config.NumSamples <= 0 || config.SeqLength <= 0 {
		return nil, errors.New("invalid configuration: samples and sequence length must be positive")
	}
	if config.PredictSteps <= 0 {
		config.PredictSteps = 1
	}

	// Generate sufficient Fibonacci numbers
	maxNeeded := config.SeqLength + config.PredictSteps + 100 // Extra buffer
	fibNumbers := make([]int, maxNeeded)
	fibNumbers[0] = 0
	if maxNeeded > 1 {
		fibNumbers[1] = 1
	}

	for i := 2; i < maxNeeded; i++ {
		next := fibNumbers[i-1] + fibNumbers[i-2]
		if config.MaxValue > 0 && next > config.MaxValue {
			maxNeeded = i
			break
		}
		fibNumbers[i] = next
	}

	if maxNeeded < config.SeqLength+config.PredictSteps {
		return nil, fmt.Errorf("cannot generate sequences of length %d with max value %d", config.SeqLength+config.PredictSteps, config.MaxValue)
	}

	sequences := make([][][]float64, config.NumSamples)
	targets := make([][]float64, config.NumSamples)

	// Find normalization factor if needed
	normFactor := 1.0
	if config.Normalize {
		maxVal := float64(fibNumbers[maxNeeded-1])
		if maxVal > 0 {
			normFactor = 1.0 / maxVal
		}
	}

	for i := 0; i < config.NumSamples; i++ {
		// Random starting position in Fibonacci sequence
		startPos := rand.Intn(maxNeeded - config.SeqLength - config.PredictSteps) //nolint:gosec // Educational implementation

		sequence := make([][]float64, config.SeqLength)
		target := make([]float64, config.PredictSteps)

		// Generate input sequence
		for t := 0; t < config.SeqLength; t++ {
			value := float64(fibNumbers[startPos+t])
			if config.Normalize {
				value *= normFactor
			}
			sequence[t] = []float64{value}
		}

		// Generate target sequence
		for p := 0; p < config.PredictSteps; p++ {
			targetValue := float64(fibNumbers[startPos+config.SeqLength+p])
			if config.Normalize {
				targetValue *= normFactor
			}
			target[p] = targetValue
		}

		sequences[i] = sequence
		targets[i] = target
	}

	return &TimeSeriesDataset{
		Name:        "Fibonacci",
		Type:        "regression",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  config.PredictSteps,
		SeqLength:   config.SeqLength,
		Description: fmt.Sprintf("Fibonacci sequence dataset: max_value=%d, normalized=%t", config.MaxValue, config.Normalize),
	}, nil
}

// GenerateRandomWalk creates a random walk time series dataset
// Mathematical Foundation: X(t) = X(t-1) + drift + ε, where ε ~ N(0, σ²)
// Learning Goal: Understanding stochastic process modeling and trend learning
func GenerateRandomWalk(config RandomWalkConfig) (*TimeSeriesDataset, error) {
	if config.NumSamples <= 0 || config.SeqLength <= 0 {
		return nil, errors.New("invalid configuration: samples and sequence length must be positive")
	}

	sequences := make([][][]float64, config.NumSamples)
	targets := make([][]float64, config.NumSamples)

	for i := 0; i < config.NumSamples; i++ {
		sequence := make([][]float64, config.SeqLength)

		// Start each walk at the configured starting value
		currentValue := config.StartValue

		// Generate random walk sequence
		for t := 0; t < config.SeqLength; t++ {
			sequence[t] = []float64{currentValue}

			// Update for next step: X(t+1) = X(t) + drift + noise
			step := rand.NormFloat64() * config.StepSize //nolint:gosec // Educational implementation
			currentValue += config.Drift + step
		}

		// Target is the next value in the walk (single-step prediction)
		nextStep := rand.NormFloat64() * config.StepSize //nolint:gosec // Educational implementation
		nextValue := currentValue + config.Drift + nextStep
		targets[i] = []float64{nextValue}

		sequences[i] = sequence
	}

	return &TimeSeriesDataset{
		Name:        "RandomWalk",
		Type:        "regression",
		Sequences:   sequences,
		Targets:     targets,
		Features:    1,
		OutputSize:  1,
		SeqLength:   config.SeqLength,
		Description: fmt.Sprintf("Random walk dataset: drift=%.3f, step_size=%.3f", config.Drift, config.StepSize),
	}, nil
}

// GetBatch returns a batch of sequences and targets
// Learning Goal: Understanding batch processing for sequence learning
func (dataset *TimeSeriesDataset) GetBatch(indices []int) ([][][]float64, [][]float64, error) {
	if len(indices) == 0 {
		return nil, nil, errors.New("empty indices")
	}

	batchSequences := make([][][]float64, len(indices))
	batchTargets := make([][]float64, len(indices))

	for i, idx := range indices {
		if idx < 0 || idx >= len(dataset.Sequences) {
			return nil, nil, fmt.Errorf("index %d out of range [0, %d)", idx, len(dataset.Sequences))
		}
		batchSequences[i] = dataset.Sequences[idx]
		batchTargets[i] = dataset.Targets[idx]
	}

	return batchSequences, batchTargets, nil
}

// Shuffle randomly shuffles dataset indices
// Learning Goal: Understanding data shuffling importance in sequence learning
func (dataset *TimeSeriesDataset) Shuffle() []int {
	indices := make([]int, len(dataset.Sequences))
	for i := range indices {
		indices[i] = i
	}

	// Fisher-Yates shuffle
	for i := len(indices) - 1; i > 0; i-- {
		j := rand.Intn(i + 1) //nolint:gosec // Educational implementation
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices
}

// GetStatistics calculates basic statistics for the time series dataset
// Learning Goal: Understanding time series data distribution analysis
func (dataset *TimeSeriesDataset) GetStatistics() map[string]float64 {
	if len(dataset.Sequences) == 0 {
		return map[string]float64{}
	}

	var allValues []float64
	var allTargets []float64

	// Collect all sequence values
	for _, sequence := range dataset.Sequences {
		for _, timestep := range sequence {
			allValues = append(allValues, timestep...)
		}
	}

	// Collect all target values
	for _, target := range dataset.Targets {
		allTargets = append(allTargets, target...)
	}

	// Calculate statistics for input sequences
	seqStats := calculateStats(allValues)
	targetStats := calculateStats(allTargets)

	return map[string]float64{
		"samples":     float64(len(dataset.Sequences)),
		"seq_length":  float64(dataset.SeqLength),
		"features":    float64(dataset.Features),
		"output_size": float64(dataset.OutputSize),
		"seq_mean":    seqStats["mean"],
		"seq_std":     seqStats["std"],
		"seq_min":     seqStats["min"],
		"seq_max":     seqStats["max"],
		"target_mean": targetStats["mean"],
		"target_std":  targetStats["std"],
		"target_min":  targetStats["min"],
		"target_max":  targetStats["max"],
	}
}

// calculateStats helper function to calculate basic statistics
func calculateStats(values []float64) map[string]float64 {
	if len(values) == 0 {
		return map[string]float64{
			"mean": 0, "std": 0, "min": 0, "max": 0,
		}
	}

	sum := 0.0
	min := values[0]
	max := values[0]

	for _, val := range values {
		sum += val
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	mean := sum / float64(len(values))

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, val := range values {
		diff := val - mean
		sumSquaredDiff += diff * diff
	}
	variance := sumSquaredDiff / float64(len(values))
	std := math.Sqrt(variance)

	return map[string]float64{
		"mean": mean,
		"std":  std,
		"min":  min,
		"max":  max,
	}
}

// PrintDatasetInfo displays comprehensive time series dataset information
// Learning Goal: Understanding time series dataset analysis and characteristics
func (dataset *TimeSeriesDataset) PrintDatasetInfo() {
	fmt.Printf("📊 Time Series Dataset: %s\n", dataset.Name)
	fmt.Printf("   📝 Description: %s\n", dataset.Description)
	fmt.Printf("   📊 Type: %s\n", dataset.Type)
	fmt.Printf("   📈 Samples: %d\n", len(dataset.Sequences))
	fmt.Printf("   ⏱️  Sequence Length: %d\n", dataset.SeqLength)
	fmt.Printf("   📐 Features per timestep: %d\n", dataset.Features)
	fmt.Printf("   🎯 Output size: %d\n", dataset.OutputSize)

	stats := dataset.GetStatistics()
	fmt.Printf("   📊 Input Statistics:\n")
	fmt.Printf("      Mean: %.4f, Std: %.4f\n", stats["seq_mean"], stats["seq_std"])
	fmt.Printf("      Range: [%.4f, %.4f]\n", stats["seq_min"], stats["seq_max"])
	fmt.Printf("   🎯 Target Statistics:\n")
	fmt.Printf("      Mean: %.4f, Std: %.4f\n", stats["target_mean"], stats["target_std"])
	fmt.Printf("      Range: [%.4f, %.4f]\n", stats["target_min"], stats["target_max"])
}

// SplitDataset splits the dataset into training and validation sets
// Learning Goal: Understanding train/validation splits for time series
func (dataset *TimeSeriesDataset) SplitDataset(trainRatio float64) (*TimeSeriesDataset, *TimeSeriesDataset, error) {
	if trainRatio <= 0 || trainRatio >= 1 {
		return nil, nil, errors.New("train ratio must be between 0 and 1")
	}

	totalSamples := len(dataset.Sequences)
	trainSize := int(float64(totalSamples) * trainRatio)

	// Ensure we have at least one sample in each set
	if trainSize == 0 {
		trainSize = 1
	}
	if trainSize >= totalSamples {
		trainSize = totalSamples - 1
	}

	// Create training dataset
	trainDataset := &TimeSeriesDataset{
		Name:        dataset.Name + "_train",
		Type:        dataset.Type,
		Sequences:   dataset.Sequences[:trainSize],
		Targets:     dataset.Targets[:trainSize],
		Features:    dataset.Features,
		OutputSize:  dataset.OutputSize,
		SeqLength:   dataset.SeqLength,
		Description: dataset.Description + " (training split)",
	}

	// Create validation dataset
	validDataset := &TimeSeriesDataset{
		Name:        dataset.Name + "_valid",
		Type:        dataset.Type,
		Sequences:   dataset.Sequences[trainSize:],
		Targets:     dataset.Targets[trainSize:],
		Features:    dataset.Features,
		OutputSize:  dataset.OutputSize,
		SeqLength:   dataset.SeqLength,
		Description: dataset.Description + " (validation split)",
	}

	return trainDataset, validDataset, nil
}

// SetRandomSeed sets the random seed for reproducible dataset generation
// Learning Goal: Understanding reproducibility in machine learning experiments
func SetRandomSeed(seed int64) {
	rand.Seed(seed) //nolint:staticcheck // Educational implementation, using legacy rand for simplicity
}

// init initializes the random seed with current time
func init() {
	rand.Seed(time.Now().UnixNano()) //nolint:staticcheck // Educational implementation
}
