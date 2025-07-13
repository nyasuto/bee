// Package data provides comprehensive tests for CSV data loading
// Learning Goal: Understanding data access layer testing patterns
package data

import (
	"os"
	"reflect"
	"testing"
)

// TestCSVDataLoader tests the CSV data loader implementation
func TestCSVDataLoader(t *testing.T) {
	loader := NewCSVDataLoader()

	t.Run("ValidCSVFile", func(t *testing.T) {
		// Create test CSV file in current directory (security restriction)
		csvContent := `1,0,1
0,1,1
0,0,0
1,1,0`
		csvFile := "test_valid.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		inputs, targets, err := loader.LoadTrainingData(csvFile)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			return
		}

		expectedInputs := [][]float64{{1, 0}, {0, 1}, {0, 0}, {1, 1}}
		expectedTargets := []float64{1, 1, 0, 0}

		if !reflect.DeepEqual(inputs, expectedInputs) {
			t.Errorf("Expected inputs %v, got %v", expectedInputs, inputs)
		}
		if !reflect.DeepEqual(targets, expectedTargets) {
			t.Errorf("Expected targets %v, got %v", expectedTargets, targets)
		}
	})

	t.Run("FloatingPointData", func(t *testing.T) {
		csvContent := `1.5,2.3,1
-0.5,1.2,0`
		csvFile := "test_float.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		inputs, targets, err := loader.LoadTrainingData(csvFile)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			return
		}

		expectedInputs := [][]float64{{1.5, 2.3}, {-0.5, 1.2}}
		expectedTargets := []float64{1, 0}

		if !reflect.DeepEqual(inputs, expectedInputs) {
			t.Errorf("Expected inputs %v, got %v", expectedInputs, inputs)
		}
		if !reflect.DeepEqual(targets, expectedTargets) {
			t.Errorf("Expected targets %v, got %v", expectedTargets, targets)
		}
	})

	t.Run("NonExistentFile", func(t *testing.T) {
		_, _, err := loader.LoadTrainingData("nonexistent.csv")
		if err == nil {
			t.Error("Expected error for non-existent file")
		}
	})

	t.Run("InsufficientColumns", func(t *testing.T) {
		csvContent := `1
0,1`
		csvFile := "test_insufficient.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		_, _, err = loader.LoadTrainingData(csvFile)
		if err == nil {
			t.Error("Expected error for insufficient columns")
		}
	})

	t.Run("InvalidNumbers", func(t *testing.T) {
		csvContent := `1,abc,1
0,1,0`
		csvFile := "test_invalid.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		_, _, err = loader.LoadTrainingData(csvFile)
		if err == nil {
			t.Error("Expected error for invalid numbers")
		}
	})

	t.Run("EmptyFile", func(t *testing.T) {
		csvFile := "test_empty.csv"
		err := os.WriteFile(csvFile, []byte(""), 0600)
		if err != nil {
			t.Fatalf("Failed to create empty CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		inputs, targets, err := loader.LoadTrainingData(csvFile)
		if err != nil {
			t.Logf("Empty file handling: %v", err)
		}
		if len(inputs) != 0 || len(targets) != 0 {
			t.Error("Expected empty results for empty file")
		}
	})
}

// TestCSVPathValidation tests path validation functionality
func TestCSVPathValidation(t *testing.T) {
	loader := NewCSVDataLoader()

	t.Run("ValidRelativePath", func(t *testing.T) {
		err := loader.ValidatePath("test.csv")
		if err != nil {
			t.Errorf("Expected valid relative path to pass, got error: %v", err)
		}
	})

	t.Run("DirectoryTraversal", func(t *testing.T) {
		err := loader.ValidatePath("../test.csv")
		if err == nil {
			t.Error("Expected error for directory traversal attempt")
		}
	})

	t.Run("AbsolutePath", func(t *testing.T) {
		err := loader.ValidatePath("/tmp/test.csv")
		if err == nil {
			t.Error("Expected error for absolute path")
		}
	})

	t.Run("EmptyPath", func(t *testing.T) {
		err := loader.ValidatePath("")
		if err != nil {
			t.Errorf("Unexpected error for empty path: %v", err)
		}
	})

	t.Run("DotDotPath", func(t *testing.T) {
		err := loader.ValidatePath("./../../test.csv")
		if err == nil {
			t.Error("Expected error for complex directory traversal")
		}
	})
}

// TestDataValidator tests data validation functionality
func TestDataValidator(t *testing.T) {
	validator := NewDefaultDataValidator()

	t.Run("ValidTrainingData", func(t *testing.T) {
		inputs := [][]float64{{1, 0}, {0, 1}, {0, 0}, {1, 1}}
		targets := []float64{1, 1, 0, 0}

		err := validator.ValidateTrainingData(inputs, targets)
		if err != nil {
			t.Errorf("Expected valid training data to pass, got error: %v", err)
		}
	})

	t.Run("MismatchedLengths", func(t *testing.T) {
		inputs := [][]float64{{1, 0}, {0, 1}}
		targets := []float64{1, 1, 0} // Extra target

		err := validator.ValidateTrainingData(inputs, targets)
		if err == nil {
			t.Error("Expected error for mismatched input/target lengths")
		}
	})

	t.Run("EmptyInputs", func(t *testing.T) {
		var inputs [][]float64
		var targets []float64

		err := validator.ValidateTrainingData(inputs, targets)
		if err == nil {
			t.Error("Expected error for empty training data")
		}
	})

	t.Run("InconsistentInputSize", func(t *testing.T) {
		inputs := [][]float64{{1, 0}, {0}} // Second input has different size
		targets := []float64{1, 0}

		err := validator.ValidateTrainingData(inputs, targets)
		if err == nil {
			t.Error("Expected error for inconsistent input sizes")
		}
	})

	t.Run("ValidInputData", func(t *testing.T) {
		input := []float64{1.0, 0.5, -0.3}

		err := validator.ValidateInputData(input)
		if err != nil {
			t.Errorf("Expected valid input data to pass, got error: %v", err)
		}
	})

	t.Run("EmptyInputData", func(t *testing.T) {
		var input []float64

		err := validator.ValidateInputData(input)
		if err == nil {
			t.Error("Expected error for empty input data")
		}
	})
}
