// Package config defines configuration structures for different commands
package config

// BaseConfig contains common configuration fields
type BaseConfig struct {
	Command string
	Verbose bool
}

// TrainConfig contains configuration for the train command
type TrainConfig struct {
	BaseConfig
	Model        string
	DataPath     string
	ModelPath    string
	LearningRate float64
	Epochs       int
}

// InferConfig contains configuration for the infer command
type InferConfig struct {
	BaseConfig
	ModelPath string
	InputData string
}

// TestConfig contains configuration for the test command
type TestConfig struct {
	BaseConfig
	Model     string
	DataPath  string
	ModelPath string
}

// BenchmarkConfig contains configuration for the benchmark command
type BenchmarkConfig struct {
	BaseConfig
	Model      string
	Dataset    string
	Iterations int
	OutputPath string
	MLPHidden  string
}

// CompareConfig contains configuration for the compare command
type CompareConfig struct {
	BaseConfig
	Dataset    string
	Iterations int
	OutputPath string
	MLPHidden  string
}

// MnistConfig contains configuration for the mnist command
type MnistConfig struct {
	BaseConfig
	DataPath string
}

// TimeSeriesConfig contains configuration for the timeseries command
type TimeSeriesConfig struct {
	BaseConfig
	Dataset string // sine, fibonacci, randomwalk
	Model   string // RNN, LSTM
	Compare bool   // Run comparison mode
}
