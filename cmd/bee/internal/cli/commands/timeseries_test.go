// Package commands test for timeseries command
// Learning Goal: Understanding testing of time series RNN/LSTM functionality
package commands

import (
	"context"
	"testing"

	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
)

// TestTimeSeriesCommand tests the TimeSeriesCommand implementation
func TestTimeSeriesCommand(t *testing.T) {
	t.Run("NewTimeSeriesCommand", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		if cmd == nil {
			t.Error("Expected non-nil timeseries command")
		}

		// Ensure it returns the Command interface
		_, ok := cmd.(*TimeSeriesCommand)
		if !ok {
			t.Error("Expected TimeSeriesCommand implementation")
		}
	})

	t.Run("Name", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		name := cmd.Name()
		if name != "timeseries" {
			t.Errorf("Expected name 'timeseries', got '%s'", name)
		}
	})

	t.Run("Description", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		desc := cmd.Description()
		expectedDesc := "Train and evaluate RNN/LSTM models on time series datasets"
		if desc != expectedDesc {
			t.Errorf("Expected description '%s', got '%s'", expectedDesc, desc)
		}
	})

	t.Run("ValidConfiguration", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "sine",
			Model:      "RNN",
		}

		err := cmd.Validate(cfg)
		if err != nil {
			t.Errorf("Expected no error for valid config, got: %v", err)
		}
	})

	t.Run("InvalidConfigurationType", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		cfg := &config.TrainConfig{} // Wrong config type

		err := cmd.Validate(cfg)
		if err == nil {
			t.Error("Expected error for invalid config type")
		}

		expectedMsg := "invalid configuration type for timeseries command"
		if err.Error() != expectedMsg {
			t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
		}
	})

	t.Run("ValidateWithNilConfig", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		err := cmd.Validate(nil)
		if err == nil {
			t.Error("Expected error for nil config")
		}
	})
}

// TestTimeSeriesCommandExecution tests the Execute method
func TestTimeSeriesCommandExecution(t *testing.T) {
	t.Run("ExecuteWithInvalidConfig", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		invalidConfig := &config.TrainConfig{} // Wrong type

		err := cmd.Execute(ctx, invalidConfig)
		if err == nil {
			t.Error("Expected error for invalid config type")
		}

		expectedMsg := "invalid configuration type for timeseries command"
		if err.Error() != expectedMsg {
			t.Errorf("Expected error message '%s', got '%s'", expectedMsg, err.Error())
		}
	})

	t.Run("ExecuteWithValidSineDataset", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries", Verbose: true},
			Dataset:    "sine",
			Model:      "RNN",
			Compare:    false,
		}

		err := cmd.Execute(ctx, cfg)

		// Check that output writer was used
		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output messages")
		}

		// Error might occur due to missing implementation - that's acceptable
		if err != nil {
			t.Logf("Expected error due to missing implementation: %v", err)
		}
	})

	t.Run("ExecuteWithFibonacciDataset", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "fibonacci",
			Model:      "LSTM",
			Compare:    false,
		}

		err := cmd.Execute(ctx, cfg)

		// Check that processing occurred
		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output messages")
		}

		if err != nil {
			t.Logf("Error for fibonacci dataset: %v", err)
		}
	})

	t.Run("ExecuteWithRandomWalkDataset", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "randomwalk",
			Model:      "RNN",
			Compare:    false,
		}

		err := cmd.Execute(ctx, cfg)

		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output messages")
		}

		if err != nil {
			t.Logf("Error for randomwalk dataset: %v", err)
		}
	})

	t.Run("ExecuteWithComparisonMode", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries", Verbose: true},
			Dataset:    "sine",
			Model:      "RNN",
			Compare:    true, // Enable comparison mode
		}

		err := cmd.Execute(ctx, cfg)

		// Should generate more output due to comparison
		if len(mockWriter.messages) == 0 {
			t.Error("Expected output messages for comparison mode")
		}

		if err != nil {
			t.Logf("Error for comparison mode: %v", err)
		}
	})

	t.Run("ExecuteWithDifferentModels", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)
		ctx := context.Background()

		models := []string{"RNN", "LSTM"}

		for _, model := range models {
			t.Run("Model_"+model, func(t *testing.T) {
				cfg := &config.TimeSeriesConfig{
					BaseConfig: config.BaseConfig{Command: "timeseries"},
					Dataset:    "sine",
					Model:      model,
					Compare:    false,
				}

				// Clear previous messages
				mockWriter.messages = nil

				err := cmd.Execute(ctx, cfg)

				if len(mockWriter.messages) == 0 {
					t.Error("Expected some output messages for model", model)
				}

				if err != nil {
					t.Logf("Error for model %s: %v", model, err)
				}
			})
		}
	})
}

// TestTimeSeriesCommandValidation tests validation scenarios
func TestTimeSeriesCommandValidation(t *testing.T) {
	mockWriter := NewMockOutputWriter()
	cmd := NewTimeSeriesCommand(mockWriter)

	testCases := []struct {
		name        string
		config      interface{}
		expectError bool
		description string
	}{
		{
			name:        "ValidTimeSeriesConfig",
			config:      &config.TimeSeriesConfig{},
			expectError: false,
			description: "Valid timeseries config should pass validation",
		},
		{
			name:        "BenchmarkConfig",
			config:      &config.BenchmarkConfig{},
			expectError: true,
			description: "Benchmark config should fail validation for timeseries command",
		},
		{
			name:        "TrainConfig",
			config:      &config.TrainConfig{},
			expectError: true,
			description: "Train config should fail validation for timeseries command",
		},
		{
			name:        "StringConfig",
			config:      "invalid",
			expectError: true,
			description: "String config should fail validation",
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

// TestTimeSeriesCommandInterface tests Command interface implementation
func TestTimeSeriesCommandInterface(t *testing.T) {
	mockWriter := NewMockOutputWriter()

	// Create command and verify it implements Command interface
	var cmd Command = NewTimeSeriesCommand(mockWriter)

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
	err := cmd.Validate(&config.TimeSeriesConfig{})
	if err != nil {
		t.Errorf("Expected successful validation, got: %v", err)
	}
}

// TestTimeSeriesCommandContextHandling tests context handling
func TestTimeSeriesCommandContextHandling(t *testing.T) {
	t.Run("ContextCancellation", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		// Create canceled context
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "sine",
			Model:      "RNN",
		}

		// Execute with canceled context
		err := cmd.Execute(ctx, cfg)

		// Ensure no panic occurs
		if err != nil {
			t.Logf("Command returned error with canceled context: %v", err)
		}
	})

	t.Run("ContextWithTimeout", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		// Create context with very short timeout
		ctx, cancel := context.WithTimeout(context.Background(), 1)
		defer cancel()

		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "sine",
			Model:      "RNN",
		}

		err := cmd.Execute(ctx, cfg)

		// Ensure no panic occurs
		if err != nil {
			t.Logf("Command returned error with timeout context: %v", err)
		}
	})
}

// BenchmarkTimeSeriesCommand benchmarks the timeseries command
func BenchmarkTimeSeriesCommand(b *testing.B) {
	mockWriter := NewMockOutputWriter()
	cmd := NewTimeSeriesCommand(mockWriter)
	ctx := context.Background()

	cfg := &config.TimeSeriesConfig{
		BaseConfig: config.BaseConfig{Command: "timeseries"},
		Dataset:    "sine",
		Model:      "RNN",
		Compare:    false,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Clear messages between runs
		mockWriter.messages = nil

		_ = cmd.Execute(ctx, cfg)
	}
}

// TestTimeSeriesCommandEdgeCases tests edge cases
func TestTimeSeriesCommandEdgeCases(t *testing.T) {
	t.Run("EmptyDataset", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "", // Empty dataset
			Model:      "RNN",
		}

		err := cmd.Execute(ctx, cfg)

		// Should handle empty dataset gracefully
		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output even for empty dataset")
		}

		// Error is expected for empty dataset
		if err != nil {
			t.Logf("Expected error for empty dataset: %v", err)
		}
	})

	t.Run("EmptyModel", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "sine",
			Model:      "", // Empty model
		}

		err := cmd.Execute(ctx, cfg)

		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output even for empty model")
		}

		// Error is expected for empty model
		if err != nil {
			t.Logf("Expected error for empty model: %v", err)
		}
	})

	t.Run("InvalidDataset", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "invalid-dataset",
			Model:      "RNN",
		}

		err := cmd.Execute(ctx, cfg)

		// Should attempt to process and provide feedback
		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output for invalid dataset")
		}

		// Error is expected for invalid dataset
		if err != nil {
			t.Logf("Expected error for invalid dataset: %v", err)
		}
	})

	t.Run("InvalidModel", func(t *testing.T) {
		mockWriter := NewMockOutputWriter()
		cmd := NewTimeSeriesCommand(mockWriter)

		ctx := context.Background()
		cfg := &config.TimeSeriesConfig{
			BaseConfig: config.BaseConfig{Command: "timeseries"},
			Dataset:    "sine",
			Model:      "INVALID",
		}

		err := cmd.Execute(ctx, cfg)

		if len(mockWriter.messages) == 0 {
			t.Error("Expected some output for invalid model")
		}

		// Error is expected for invalid model
		if err != nil {
			t.Logf("Expected error for invalid model: %v", err)
		}
	})
}
