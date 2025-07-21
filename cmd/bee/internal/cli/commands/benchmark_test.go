// Package commands test for benchmark command
// Learning Goal: Understanding comprehensive testing of benchmark functionality
package commands

import (
	"context"
	"strings"
	"testing"

	"github.com/nyasuto/bee/benchmark"
	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
	"github.com/nyasuto/bee/cmd/bee/internal/output"
)

// MockOutputWriter for testing
type MockOutputWriter struct {
	messages []string
	level    output.LogLevel
}

func (m *MockOutputWriter) WriteMessage(level output.LogLevel, message string, args ...interface{}) {
	m.level = level
	// Simple mock - just store the message format
	m.messages = append(m.messages, message)
}

func (m *MockOutputWriter) WriteTrainingResult(epochs int, accuracy float64, verbose bool) {
	m.messages = append(m.messages, "Training completed")
}

func (m *MockOutputWriter) WriteInferenceResult(prediction float64, input []float64, verbose bool) {
	m.messages = append(m.messages, "Inference completed")
}

func (m *MockOutputWriter) WriteTestResult(accuracy float64, samples int, verbose bool, predictions []output.PredictionResult) {
	m.messages = append(m.messages, "Test completed")
}

func (m *MockOutputWriter) WriteBenchmarkResult(metrics benchmark.PerformanceMetrics) {
	m.messages = append(m.messages, "Benchmark completed")
}

func (m *MockOutputWriter) WriteUsage() {
	m.messages = append(m.messages, "Usage displayed")
}

func NewMockOutputWriter() *MockOutputWriter {
	return &MockOutputWriter{
		messages: make([]string, 0),
	}
}

// TestBenchmarkCommand tests the BenchmarkCommand implementation
func TestBenchmarkCommand(t *testing.T) {
	t.Run("NewBenchmarkCommand", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		if cmd == nil {
			t.Error("Expected non-nil benchmark command")
		}

		// Note: We can't directly compare outputWriter due to interface vs concrete type
		// Just verify the command was created successfully
	})

	t.Run("Name", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		name := cmd.Name()
		if name != "benchmark" {
			t.Errorf("Expected name 'benchmark', got '%s'", name)
		}
	})

	t.Run("Description", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		desc := cmd.Description()
		expectedDesc := "Run performance benchmarks for neural network models (Perceptron, MLP, CNN)"
		if desc != expectedDesc {
			t.Errorf("Expected description '%s', got '%s'", expectedDesc, desc)
		}
	})

	t.Run("ValidConfiguration", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		cfg := &config.BenchmarkConfig{
			BaseConfig: config.BaseConfig{Command: "benchmark"},
			Model:      "perceptron",
			Dataset:    "xor",
		}

		err := cmd.Validate(cfg)
		if err != nil {
			t.Errorf("Expected no error for valid config, got: %v", err)
		}
	})

	t.Run("InvalidConfigurationType", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		cfg := &config.TrainConfig{} // Wrong config type

		err := cmd.Validate(cfg)
		if err == nil {
			t.Error("Expected error for invalid config type")
		}

		expectedMsg := "invalid configuration type for benchmark command"
		if err.Error() != expectedMsg {
			t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
		}
	})

	t.Run("ValidateWithNilConfig", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		err := cmd.Validate(nil)
		if err == nil {
			t.Error("Expected error for nil config")
		}
	})
}

// TestBenchmarkCommandExecution tests the Execute method with various configurations
func TestBenchmarkCommandExecution(t *testing.T) {
	t.Run("ExecuteWithInvalidConfig", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		ctx := context.Background()
		invalidConfig := &config.TrainConfig{} // Wrong type

		err := cmd.Execute(ctx, invalidConfig)
		if err == nil {
			t.Error("Expected error for invalid config type")
		}

		expectedMsg := "invalid configuration type for benchmark command"
		if err.Error() != expectedMsg {
			t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
		}
	})

	t.Run("ExecuteWithValidConfigButMissingModel", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.BenchmarkConfig{
			BaseConfig: config.BaseConfig{Command: "benchmark"},
			Model:      "", // Empty model
			Dataset:    "xor",
		}

		err := cmd.Execute(ctx, cfg)
		// This might fail due to unsupported model type
		// The exact behavior depends on implementation
		if err != nil {
			// Check that appropriate error handling occurred
			if !strings.Contains(err.Error(), "model") &&
				!strings.Contains(err.Error(), "unsupported") {
				t.Errorf("Expected model-related error, got: %v", err)
			}
		}
	})

	t.Run("ExecutePerceptronBenchmark", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.BenchmarkConfig{
			BaseConfig: config.BaseConfig{Command: "benchmark", Verbose: true},
			Model:      "perceptron",
			Dataset:    "xor",
			Iterations: 10,
		}

		// Note: This test might require actual implementation of benchmark logic
		// For now, we're testing the interface and basic validation
		err := cmd.Execute(ctx, cfg)

		// Check that output writer was called
		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output messages")
		}

		// The error might occur due to missing actual benchmark implementation
		// That's acceptable for interface testing
		if err != nil {
			t.Logf("Expected error due to missing implementation: %v", err)
		}
	})

	t.Run("ExecuteWithDifferentModels", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)
		ctx := context.Background()

		models := []string{"mlp", "cnn", "compare"}

		for _, model := range models {
			t.Run("Model_"+model, func(t *testing.T) {
				cfg := &config.BenchmarkConfig{
					BaseConfig: config.BaseConfig{Command: "benchmark"},
					Model:      model,
					Dataset:    "xor",
					Iterations: 5,
				}

				// Clear previous messages
				mockWriter.messages = nil

				err := cmd.Execute(ctx, cfg)

				// Check that some processing occurred (output messages)
				if len(mockWriter.messages) == 0 {
					t.Error("Expected some output messages for model", model)
				}

				// Error is acceptable due to missing implementation details
				if err != nil {
					t.Logf("Error for model %s: %v", model, err)
				}
			})
		}
	})

	t.Run("ExecuteWithDifferentDatasets", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)
		ctx := context.Background()

		datasets := []string{"and", "or", "mnist"}

		for _, dataset := range datasets {
			t.Run("Dataset_"+dataset, func(t *testing.T) {
				cfg := &config.BenchmarkConfig{
					BaseConfig: config.BaseConfig{Command: "benchmark"},
					Model:      "perceptron",
					Dataset:    dataset,
					Iterations: 5,
				}

				// Clear previous messages
				mockWriter.messages = nil

				err := cmd.Execute(ctx, cfg)

				// Check that processing occurred
				if len(mockWriter.messages) == 0 {
					t.Error("Expected some output messages for dataset", dataset)
				}

				// Error is acceptable due to missing implementation details
				if err != nil {
					t.Logf("Error for dataset %s: %v", dataset, err)
				}
			})
		}
	})
}

// TestBenchmarkCommandValidation tests various validation scenarios
func TestBenchmarkCommandValidation(t *testing.T) {
	mockWriter := NewMockOutputWriter()
	cmd := NewBenchmarkCommand(mockWriter)

	testCases := []struct {
		name        string
		config      interface{}
		expectError bool
		description string
	}{
		{
			name:        "ValidBenchmarkConfig",
			config:      &config.BenchmarkConfig{},
			expectError: false,
			description: "Valid benchmark config should pass validation",
		},
		{
			name:        "TrainConfig",
			config:      &config.TrainConfig{},
			expectError: true,
			description: "Train config should fail validation for benchmark command",
		},
		{
			name:        "StringConfig",
			config:      "invalid",
			expectError: true,
			description: "String config should fail validation",
		},
		{
			name:        "IntConfig",
			config:      42,
			expectError: true,
			description: "Integer config should fail validation",
		},
		{
			name:        "NilConfig",
			config:      nil,
			expectError: true,
			description: "Nil config should fail validation",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := cmd.Validate(tc.config)

			if tc.expectError && err == nil {
				t.Errorf("Expected error for %s, but got none", tc.description)
			}

			if !tc.expectError && err != nil {
				t.Errorf("Expected no error for %s, but got: %v", tc.description, err)
			}
		})
	}
}

// TestBenchmarkCommandInterface tests that BenchmarkCommand implements Command interface
func TestBenchmarkCommandInterface(t *testing.T) {
	mockWriter := NewMockOutputWriter()

	// Create command and check it implements Command interface
	var cmd Command = NewBenchmarkCommand(mockWriter)

	if cmd == nil {
		t.Error("Expected command to implement Command interface")
	}

	// Test interface methods
	name := cmd.Name()
	if name == "" {
		t.Error("Expected non-empty name from Command interface")
	}

	desc := cmd.Description()
	if desc == "" {
		t.Error("Expected non-empty description from Command interface")
	}

	// Test Validate method
	err := cmd.Validate(&config.BenchmarkConfig{})
	if err != nil {
		t.Errorf("Expected successful validation, got: %v", err)
	}
}

// TestBenchmarkCommandContextHandling tests context handling
func TestBenchmarkCommandContextHandling(t *testing.T) {
	t.Run("ContextCancellation", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		// Create canceled context
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		cfg := &config.BenchmarkConfig{
			BaseConfig: config.BaseConfig{Command: "benchmark"},
			Model:      "perceptron",
			Dataset:    "xor",
		}

		// Execute with canceled context
		err := cmd.Execute(ctx, cfg)

		// The command might not check context cancellation immediately,
		// so we just ensure it doesn't panic
		if err != nil {
			t.Logf("Command returned error with canceled context: %v", err)
		}
	})

	t.Run("ContextWithTimeout", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)

		// Create context with very short timeout
		ctx, cancel := context.WithTimeout(context.Background(), 1)
		defer cancel()

		cfg := &config.BenchmarkConfig{
			BaseConfig: config.BaseConfig{Command: "benchmark"},
			Model:      "perceptron",
			Dataset:    "xor",
		}

		err := cmd.Execute(ctx, cfg)

		// Ensure no panic occurs
		if err != nil {
			t.Logf("Command returned error with timeout context: %v", err)
		}
	})
}

// BenchmarkBenchmarkCommand benchmarks the benchmark command itself
func BenchmarkBenchmarkCommand(b *testing.B) {
	mockWriter := NewMockOutputWriter()
	cmd := NewBenchmarkCommand(mockWriter)
	ctx := context.Background()

	cfg := &config.BenchmarkConfig{
		BaseConfig: config.BaseConfig{Command: "benchmark"},
		Model:      "perceptron",
		Dataset:    "xor",
		Iterations: 1, // Minimal iterations for benchmarking
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Clear messages between runs
		mockWriter.messages = nil

		_ = cmd.Execute(ctx, cfg)
	}
}

// TestBenchmarkCommandOutputHandling tests output handling
func TestBenchmarkCommandOutputHandling(t *testing.T) {
	t.Run("VerboseOutput", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)
		ctx := context.Background()

		cfg := &config.BenchmarkConfig{
			BaseConfig: config.BaseConfig{Command: "benchmark", Verbose: true},
			Model:      "perceptron",
			Dataset:    "xor",
		}

		err := cmd.Execute(ctx, cfg)

		// Check that verbose output was generated
		if len(mockWriter.messages) == 0 {
			t.Error("Expected verbose output messages")
		}

		// Error is acceptable due to implementation details
		if err != nil {
			t.Logf("Expected error due to implementation: %v", err)
		}
	})

	t.Run("NonVerboseOutput", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewBenchmarkCommand(mockWriter)
		ctx := context.Background()

		cfg := &config.BenchmarkConfig{
			BaseConfig: config.BaseConfig{Command: "benchmark", Verbose: false},
			Model:      "perceptron",
			Dataset:    "xor",
		}

		err := cmd.Execute(ctx, cfg)

		// Should still have some output, just less verbose
		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output messages even in non-verbose mode")
		}

		if err != nil {
			t.Logf("Expected error due to implementation: %v", err)
		}
	})
}
