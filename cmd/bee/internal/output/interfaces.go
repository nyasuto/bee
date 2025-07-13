// Package output defines interfaces for output operations
package output

import "github.com/nyasuto/bee/benchmark"

// OutputWriter handles different types of output
type OutputWriter interface {
	// WriteMessage writes a simple message
	WriteMessage(level LogLevel, message string, args ...interface{})

	// WriteTrainingResult writes training results
	WriteTrainingResult(epochs int, accuracy float64, verbose bool)

	// WriteInferenceResult writes inference results
	WriteInferenceResult(prediction float64, input []float64, verbose bool)

	// WriteTestResult writes test results
	WriteTestResult(accuracy float64, samples int, verbose bool, predictions []PredictionResult)

	// WriteBenchmarkResult writes benchmark results
	WriteBenchmarkResult(metrics benchmark.PerformanceMetrics)

	// WriteUsage writes usage information
	WriteUsage()
}

// LogLevel represents the level of log messages
type LogLevel int

const (
	LogLevelInfo LogLevel = iota
	LogLevelWarn
	LogLevelError
	LogLevelSuccess
)

// PredictionResult represents a single prediction result for verbose output
type PredictionResult struct {
	Input     []float64
	Predicted float64
	Actual    float64
	Correct   bool
}

// Logger provides structured logging capabilities
type Logger interface {
	// Info logs an informational message
	Info(message string, args ...interface{})

	// Warn logs a warning message
	Warn(message string, args ...interface{})

	// Error logs an error message
	Error(message string, args ...interface{})

	// Success logs a success message
	Success(message string, args ...interface{})
}
