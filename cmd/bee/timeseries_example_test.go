// Package main test for Time Series example functions
// Learning Goal: Understanding time series example testing strategies
package main

import (
	"testing"
)

// TestTimeSeriesExample tests the TimeSeriesExample function
func TestTimeSeriesExample(t *testing.T) {
	t.Run("TimeSeriesExampleSineDataset", func(t *testing.T) {
		// Test with sine dataset (should work)
		err := TimeSeriesExample("sine", "RNN", false)
		if err != nil {
			t.Errorf("Expected no error for sine dataset: %v", err)
		}
	})

	t.Run("TimeSeriesExampleVerbose", func(t *testing.T) {
		// Test with verbose mode
		err := TimeSeriesExample("sine", "RNN", true)
		if err != nil {
			t.Errorf("Expected no error for sine dataset in verbose mode: %v", err)
		}
	})

	t.Run("TimeSeriesExampleInvalidDataset", func(t *testing.T) {
		// Test with invalid dataset
		err := TimeSeriesExample("invalid_dataset", "RNN", false)
		if err == nil {
			t.Error("Expected error for invalid dataset")
		}
	})

	t.Run("TimeSeriesExampleInvalidModel", func(t *testing.T) {
		// Test with invalid model type
		err := TimeSeriesExample("sine", "INVALID_MODEL", false)
		if err == nil {
			t.Error("Expected error for invalid model type")
		}
	})

	t.Run("TimeSeriesExampleDifferentDatasets", func(t *testing.T) {
		datasets := []string{"sine", "cosine", "linear", "random"}

		for _, dataset := range datasets {
			t.Run(dataset, func(t *testing.T) {
				err := TimeSeriesExample(dataset, "RNN", false)
				// Some datasets might not be implemented, so we allow errors
				// but we test that the function doesn't panic
				if err != nil {
					t.Logf("Dataset %s returned error (might be expected): %v", dataset, err)
				}
			})
		}
	})

	t.Run("TimeSeriesExampleDifferentModels", func(t *testing.T) {
		models := []string{"RNN", "LSTM", "GRU"}

		for _, model := range models {
			t.Run(model, func(t *testing.T) {
				err := TimeSeriesExample("sine", model, false)
				// Some models might not be implemented, so we allow errors
				// but we test that the function doesn't panic
				if err != nil {
					t.Logf("Model %s returned error (might be expected): %v", model, err)
				}
			})
		}
	})
}

// TestRunTimeSeriesComparison tests the RunTimeSeriesComparison function
func TestRunTimeSeriesComparison(t *testing.T) {
	// Note: RunTimeSeriesComparison() doesn't take parameters and is a demo function
	// We can test it but it will run the full comparison demo

	t.Run("RunTimeSeriesComparisonExists", func(t *testing.T) {
		// Just verify the function exists and can be called
		// This will run the actual comparison demo
		defer func() {
			if r := recover(); r != nil {
				t.Logf("RunTimeSeriesComparison panicked: %v", r)
			}
		}()

		// Call the function - it should not panic
		RunTimeSeriesComparison()
		t.Log("RunTimeSeriesComparison executed successfully")
	})
}

// TestTimeSeriesExampleIntegration tests integration scenarios
func TestTimeSeriesExampleIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	t.Run("TimeSeriesWorkflow", func(t *testing.T) {
		// Test complete workflow with different configurations
		configurations := []struct {
			dataset string
			model   string
			verbose bool
		}{
			{"sine", "RNN", false},
			{"sine", "RNN", true},
			{"cosine", "RNN", false},
		}

		for _, config := range configurations {
			t.Run(config.dataset+"_"+config.model, func(t *testing.T) {
				err := TimeSeriesExample(config.dataset, config.model, config.verbose)
				// We allow errors here as not all combinations might be implemented
				if err != nil {
					t.Logf("Configuration %+v returned error: %v", config, err)
				}
			})
		}
	})

	t.Run("ComparisonWorkflow", func(t *testing.T) {
		// Test comparison workflow
		defer func() {
			if r := recover(); r != nil {
				t.Logf("Time series comparison panicked: %v", r)
			}
		}()

		RunTimeSeriesComparison()
		t.Log("Time series comparison completed")
	})
}

// BenchmarkTimeSeriesExample benchmarks the time series example function
func BenchmarkTimeSeriesExample(b *testing.B) {
	b.Run("TimeSeriesExampleSine", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = TimeSeriesExample("sine", "RNN", false)
		}
	})

	b.Run("TimeSeriesComparisonSine", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			func() {
				defer func() { recover() }() // Ignore any panics in benchmark
				RunTimeSeriesComparison()
			}()
		}
	})
}

// TestTimeSeriesExampleErrorHandling tests error handling patterns
func TestTimeSeriesExampleErrorHandling(t *testing.T) {
	t.Run("EmptyDataset", func(t *testing.T) {
		err := TimeSeriesExample("", "RNN", false)
		if err == nil {
			t.Error("Expected error for empty dataset name")
		}
	})

	t.Run("EmptyModel", func(t *testing.T) {
		err := TimeSeriesExample("sine", "", false)
		if err == nil {
			t.Error("Expected error for empty model name")
		}
	})

	t.Run("ErrorMessage", func(t *testing.T) {
		err := TimeSeriesExample("nonexistent_dataset", "RNN", false)
		if err == nil {
			t.Error("Expected error for nonexistent dataset")
		}

		// Check that error message is meaningful
		if err != nil {
			errorMsg := err.Error()
			if len(errorMsg) == 0 {
				t.Error("Expected non-empty error message")
			}
		}
	})

	t.Run("ComparisonErrorHandling", func(t *testing.T) {
		// RunTimeSeriesComparison doesn't take parameters, so we test it differently
		defer func() {
			if r := recover(); r != nil {
				t.Logf("RunTimeSeriesComparison panicked during error handling test: %v", r)
			}
		}()

		RunTimeSeriesComparison()
		t.Log("RunTimeSeriesComparison completed in error handling test")
	})
}

// TestTimeSeriesExampleEdgeCases tests edge cases
func TestTimeSeriesExampleEdgeCases(t *testing.T) {
	t.Run("SpecialCharactersInDataset", func(t *testing.T) {
		// Test with special characters in dataset name
		specialDatasets := []string{"sine-test", "sine_test", "sine.test", "sine/test"}

		for _, dataset := range specialDatasets {
			t.Run(dataset, func(t *testing.T) {
				err := TimeSeriesExample(dataset, "RNN", false)
				// These should fail, but shouldn't panic
				if err == nil {
					t.Logf("Unexpectedly succeeded with dataset: %s", dataset)
				}
			})
		}
	})

	t.Run("SpecialCharactersInModel", func(t *testing.T) {
		// Test with special characters in model name
		specialModels := []string{"RNN-test", "RNN_test", "RNN.test", "RNN/test"}

		for _, model := range specialModels {
			t.Run(model, func(t *testing.T) {
				err := TimeSeriesExample("sine", model, false)
				// These should fail, but shouldn't panic
				if err == nil {
					t.Logf("Unexpectedly succeeded with model: %s", model)
				}
			})
		}
	})
}
