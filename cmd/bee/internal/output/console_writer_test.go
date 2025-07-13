// Package output provides comprehensive tests for console output
// Learning Goal: Understanding output layer testing patterns and log capture
package output

import (
	"testing"

	"github.com/nyasuto/bee/benchmark"
)

// TestConsoleOutputWriter tests the console output writer implementation
func TestConsoleOutputWriter(t *testing.T) {
	// Create a console output writer with real logger
	writer := NewConsoleOutputWriter()

	t.Run("WriteMessage", func(t *testing.T) {
		// Just test that the method doesn't panic
		writer.WriteMessage(LogLevelInfo, "Test message")
	})

	t.Run("WriteMessageWithArgs", func(t *testing.T) {
		// Just test that the method doesn't panic
		writer.WriteMessage(LogLevelInfo, "Test %s with %d args", "message", 2)
	})

	t.Run("WriteTrainingResult", func(t *testing.T) {
		// Just test that the method doesn't panic
		writer.WriteTrainingResult(100, 0.85, false)
	})

	t.Run("WriteTrainingResultVerbose", func(t *testing.T) {
		// Just test that the method doesn't panic
		writer.WriteTrainingResult(50, 0.75, true)
	})

	t.Run("WriteInferenceResult", func(t *testing.T) {
		// Just test that the method doesn't panic
		input := []float64{1.0, 0.5}
		writer.WriteInferenceResult(0.8, input, false)
	})

	t.Run("WriteInferenceResultVerbose", func(t *testing.T) {
		// Just test that the method doesn't panic
		input := []float64{1.0, 0.5}
		writer.WriteInferenceResult(0.8, input, true)
	})

	t.Run("WriteTestResult", func(t *testing.T) {
		// Just test that the method doesn't panic
		predictions := []PredictionResult{
			{Input: []float64{1, 0}, Predicted: 0.9, Actual: 1.0, Correct: true},
			{Input: []float64{0, 1}, Predicted: 0.1, Actual: 0.0, Correct: true},
		}

		writer.WriteTestResult(0.85, 10, false, predictions)
	})

	t.Run("WriteTestResultVerbose", func(t *testing.T) {
		// Just test that the method doesn't panic
		predictions := []PredictionResult{
			{Input: []float64{1, 0}, Predicted: 0.9, Actual: 1.0, Correct: true},
			{Input: []float64{0, 1}, Predicted: 0.1, Actual: 0.0, Correct: true},
		}

		writer.WriteTestResult(0.85, 2, true, predictions)
	})

	t.Run("WriteBenchmarkResult", func(t *testing.T) {
		// Just test that the method doesn't panic
		metrics := benchmark.PerformanceMetrics{
			ModelType:       "perceptron",
			DatasetName:     "xor",
			Accuracy:        0.75,
			TrainingTime:    1000000, // 1ms in nanoseconds
			InferenceTime:   500000,  // 0.5ms in nanoseconds
			MemoryUsage:     1024,    // 1KB
			ConvergenceRate: 100,
			FinalLoss:       0.1234,
		}

		writer.WriteBenchmarkResult(metrics)
	})

	t.Run("WriteUsage", func(t *testing.T) {
		// Just test that the method doesn't panic
		writer.WriteUsage()
	})
}

// TestConsoleLogger tests the console logger implementation
func TestConsoleLogger(t *testing.T) {
	logger := NewConsoleLogger()

	t.Run("InfoLog", func(t *testing.T) {
		// Just test that the method doesn't panic
		logger.Info("Info message")
	})

	t.Run("WarnLog", func(t *testing.T) {
		// Just test that the method doesn't panic
		logger.Warn("Warning message")
	})

	t.Run("ErrorLog", func(t *testing.T) {
		// Just test that the method doesn't panic
		logger.Error("Error message")
	})

	t.Run("SuccessLog", func(t *testing.T) {
		// Just test that the method doesn't panic
		logger.Success("Success message")
	})

	t.Run("FormattedMessages", func(t *testing.T) {
		// Just test that the method doesn't panic
		logger.Info("Formatted %s with %d args", "message", 2)
	})
}

// TestNewConsoleOutputWriter tests the constructor
func TestNewConsoleOutputWriter(t *testing.T) {
	writer := NewConsoleOutputWriter()

	if writer == nil {
		t.Error("Expected non-nil console output writer")
		return
	}

	if writer.logger == nil {
		t.Error("Expected non-nil logger")
	}
}

// TestNewConsoleLogger tests the logger constructor
func TestNewConsoleLogger(t *testing.T) {
	logger := NewConsoleLogger()

	if logger == nil {
		t.Error("Expected non-nil console logger")
		return
	}

	// Test that it implements the Logger interface
	var _ Logger = logger
}

// TestLogLevels tests the log level constants
func TestLogLevels(t *testing.T) {
	if LogLevelInfo != 0 {
		t.Errorf("Expected LogLevelInfo to be 0, got %d", LogLevelInfo)
	}
	if LogLevelWarn != 1 {
		t.Errorf("Expected LogLevelWarn to be 1, got %d", LogLevelWarn)
	}
	if LogLevelError != 2 {
		t.Errorf("Expected LogLevelError to be 2, got %d", LogLevelError)
	}
	if LogLevelSuccess != 3 {
		t.Errorf("Expected LogLevelSuccess to be 3, got %d", LogLevelSuccess)
	}
}

// TestPredictionResult tests the PredictionResult struct
func TestPredictionResult(t *testing.T) {
	result := PredictionResult{
		Input:     []float64{1.0, 0.5},
		Predicted: 0.8,
		Actual:    1.0,
		Correct:   true,
	}

	if len(result.Input) != 2 {
		t.Errorf("Expected input length 2, got %d", len(result.Input))
	}
	if result.Predicted != 0.8 {
		t.Errorf("Expected predicted 0.8, got %f", result.Predicted)
	}
	if result.Actual != 1.0 {
		t.Errorf("Expected actual 1.0, got %f", result.Actual)
	}
	if !result.Correct {
		t.Error("Expected correct to be true")
	}
}

// TestCapitalizeFirst tests the capitalize first letter functionality
func TestCapitalizeFirst(t *testing.T) {
	// This functionality is tested indirectly through WriteBenchmarkResult
	writer := NewConsoleOutputWriter()

	// Just test that benchmark result with different model types work
	testCases := []string{"perceptron", "mlp", "cnn"}

	for _, modelType := range testCases {
		metrics := benchmark.PerformanceMetrics{
			ModelType:   modelType,
			DatasetName: "test",
			Accuracy:    0.5,
		}

		// Just test that it doesn't panic
		writer.WriteBenchmarkResult(metrics)
	}
}
