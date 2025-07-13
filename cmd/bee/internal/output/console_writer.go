// Package output provides output handling implementations
package output

import (
	"fmt"
	"os"
	"unicode"

	"github.com/nyasuto/bee/benchmark"
)

// ConsoleOutputWriter implements OutputWriter for console output
type ConsoleOutputWriter struct {
	logger Logger
}

// NewConsoleOutputWriter creates a new console output writer
func NewConsoleOutputWriter() *ConsoleOutputWriter {
	return &ConsoleOutputWriter{
		logger: NewConsoleLogger(),
	}
}

// WriteMessage writes a simple message to console
func (c *ConsoleOutputWriter) WriteMessage(level LogLevel, message string, args ...interface{}) {
	switch level {
	case LogLevelInfo:
		c.logger.Info(message, args...)
	case LogLevelWarn:
		c.logger.Warn(message, args...)
	case LogLevelError:
		c.logger.Error(message, args...)
	case LogLevelSuccess:
		c.logger.Success(message, args...)
	}
}

// WriteTrainingResult writes training results to console
func (c *ConsoleOutputWriter) WriteTrainingResult(epochs int, accuracy float64, verbose bool) {
	c.logger.Success("Training completed in %d epochs", epochs)
	c.logger.Info("Training accuracy: %.2f%%", accuracy*100)

	if verbose {
		c.logger.Info("Training completed with verbose output enabled")
	}
}

// WriteInferenceResult writes inference results to console
func (c *ConsoleOutputWriter) WriteInferenceResult(prediction float64, input []float64, verbose bool) {
	c.logger.Info("Prediction: %.0f", prediction)

	if verbose {
		c.logger.Info("Input: %v", input)
	}
}

// WriteTestResult writes test results to console
func (c *ConsoleOutputWriter) WriteTestResult(accuracy float64, samples int, verbose bool, predictions []PredictionResult) {
	c.logger.Info("Test Results:")
	c.logger.Info("   Samples: %d", samples)
	c.logger.Info("   Accuracy: %.2f%%", accuracy*100)

	if verbose && len(predictions) > 0 {
		c.logger.Info("\nDetailed Predictions:")
		correct := 0
		for _, pred := range predictions {
			status := "❌"
			if pred.Correct {
				status = "✅"
				correct++
			}
			c.logger.Info("   %s Input: %v → Predicted: %.0f, Actual: %.0f",
				status, pred.Input, pred.Predicted, pred.Actual)
		}
		c.logger.Info("\nSummary: %d/%d correct", correct, len(predictions))
	}
}

// WriteBenchmarkResult writes benchmark results to console
func (c *ConsoleOutputWriter) WriteBenchmarkResult(metrics benchmark.PerformanceMetrics) {
	// Capitalize first letter (replacing deprecated strings.Title)
	modelType := metrics.ModelType
	if len(modelType) > 0 {
		runes := []rune(modelType)
		runes[0] = unicode.ToUpper(runes[0])
		modelType = string(runes)
	}
	c.logger.Info("🧠 %s Model Results:", modelType)
	c.logger.Info("   Dataset: %s", metrics.DatasetName)
	c.logger.Info("   Accuracy: %.2f%%", metrics.Accuracy*100)
	c.logger.Info("   Training Time: %s", benchmark.FormatDuration(metrics.TrainingTime))
	c.logger.Info("   Inference Time: %s", benchmark.FormatDuration(metrics.InferenceTime))
	c.logger.Info("   Memory Usage: %s", benchmark.FormatMemory(metrics.MemoryUsage))
	c.logger.Info("   Convergence: %d epochs", metrics.ConvergenceRate)
	c.logger.Info("   Final Loss: %.4f", metrics.FinalLoss)
	c.logger.Info("   Timestamp: %s", metrics.Timestamp.Format("2006-01-02 15:04:05"))
	c.logger.Info("")
}

// WriteUsage writes usage information to console
func (c *ConsoleOutputWriter) WriteUsage() {
	fmt.Printf(`🐝 Bee Neural Network CLI Tool

Usage:
  bee <command> [options]

Commands:
  train      Train a neural network model
  infer      Perform inference with a trained model
  test       Test a trained model on data
  benchmark  Run performance benchmarks
  compare    Compare model performance
  mnist      MNIST CNN demonstration
  help       Show this help message

Training:
  bee train -data <csv_file> [options]
    -model string     Model type (default "perceptron")
    -data string      Path to training data (CSV format)
    -output string    Output model path (default "model.json")
    -lr float         Learning rate (default 0.1)
    -epochs int       Maximum training epochs (default 1000)
    -verbose          Verbose output

Inference:
  bee infer -model <model_file> -input <values>
    -model string     Path to trained model (default "model.json")
    -input string     Comma-separated input values
    -verbose          Verbose output

Testing:
  bee test -data <csv_file> [options]
    -data string      Path to test data (CSV format)
    -model-path string Path to trained model (default "model.json")
    -verbose          Verbose output

Benchmarking:
  bee benchmark [options]
    -model string     Model type: perceptron, mlp, both (default "perceptron")
    -dataset string   Dataset: xor, and, or, all (default "xor")
    -iterations int   Number of benchmark iterations (default 100)
    -mlp-hidden string Hidden layer sizes for MLP (default "4")
    -output string    Output file for results (JSON)
    -verbose          Verbose output

Comparison:
  bee compare [options]
    -dataset string   Dataset: xor, and, or, all (default "xor")
    -iterations int   Number of benchmark iterations (default 100)
    -mlp-hidden string Hidden layer sizes for MLP (default "4")
    -output string    Output file for results (JSON)
    -verbose          Verbose output

MNIST CNN Demo:
  bee mnist [options]
    -data-dir string  Directory for MNIST data (default "datasets/mnist")
    -verbose          Verbose output

Examples:
  # Train XOR perceptron
  bee train -data datasets/xor.csv -output models/xor.json -verbose

  # Test XOR patterns
  bee infer -model models/xor.json -input "1,1"
  bee infer -model models/xor.json -input "0,1"

  # Test model accuracy
  bee test -data datasets/xor_test.csv -model-path models/xor.json

  # Benchmark perceptron on XOR
  bee benchmark -model perceptron -dataset xor -verbose

  # Compare perceptron vs MLP on all datasets
  bee compare -dataset all -mlp-hidden "4,2" -verbose

  # Benchmark both models with custom iterations
  bee benchmark -model both -dataset xor -iterations 50

  # MNIST CNN demonstration
  bee mnist -verbose

Data Format (CSV):
  # XOR dataset example
  0,0,0
  0,1,1
  1,0,1
  1,1,0

Learning Resources:
  - Phase 1.0/1.1: Perceptron and MLP fundamentals
  - Phase 1.5: Performance measurement and comparison
  - Each command demonstrates different ML pipeline stages
  - Verbose mode shows internal weights and calculations
  - Benchmark commands help understand algorithm performance trade-offs
`)
}

// ConsoleLogger implements Logger for console output
type ConsoleLogger struct{}

// NewConsoleLogger creates a new console logger
func NewConsoleLogger() *ConsoleLogger {
	return &ConsoleLogger{}
}

// Info logs an informational message
func (c *ConsoleLogger) Info(message string, args ...interface{}) {
	fmt.Printf(message+"\n", args...)
}

// Warn logs a warning message
func (c *ConsoleLogger) Warn(message string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "⚠️  "+message+"\n", args...)
}

// Error logs an error message
func (c *ConsoleLogger) Error(message string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "❌ "+message+"\n", args...)
}

// Success logs a success message
func (c *ConsoleLogger) Success(message string, args ...interface{}) {
	fmt.Printf("✅ "+message+"\n", args...)
}
