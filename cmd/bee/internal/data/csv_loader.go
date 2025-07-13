// Package data provides data loading and validation implementations
package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// CSVDataLoader implements DataLoader for CSV files
type CSVDataLoader struct{}

// NewCSVDataLoader creates a new CSV data loader
func NewCSVDataLoader() *CSVDataLoader {
	return &CSVDataLoader{}
}

// LoadTrainingData loads training data from a CSV file
func (c *CSVDataLoader) LoadTrainingData(path string) ([][]float64, []float64, error) {
	if err := c.ValidatePath(path); err != nil {
		return nil, nil, err
	}

	// #nosec G304 - path is validated above
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot open file: %w", err)
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			// Log warning but don't override the main error
			fmt.Printf("Warning: failed to close file: %v\n", closeErr)
		}
	}()

	reader := csv.NewReader(file)
	var inputs [][]float64
	var targets []float64

	lineNum := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("error reading CSV line %d: %w", lineNum+1, err)
		}

		lineNum++

		// Skip empty lines or comments
		if len(record) == 0 || strings.HasPrefix(record[0], "#") {
			continue
		}

		if len(record) < 2 {
			return nil, nil, fmt.Errorf("line %d: insufficient columns, need at least 2", lineNum)
		}

		// Parse features (all columns except last)
		features := make([]float64, len(record)-1)
		for i := 0; i < len(record)-1; i++ {
			features[i], err = strconv.ParseFloat(strings.TrimSpace(record[i]), 64)
			if err != nil {
				return nil, nil, fmt.Errorf("line %d, column %d: invalid number '%s'",
					lineNum, i+1, record[i])
			}
		}

		// Parse target (last column)
		target, err := strconv.ParseFloat(strings.TrimSpace(record[len(record)-1]), 64)
		if err != nil {
			return nil, nil, fmt.Errorf("line %d, target column: invalid number '%s'",
				lineNum, record[len(record)-1])
		}

		inputs = append(inputs, features)
		targets = append(targets, target)
	}

	return inputs, targets, nil
}

// ValidatePath checks if the data path is valid and accessible
func (c *CSVDataLoader) ValidatePath(path string) error {
	// Prevent directory traversal and absolute paths
	if strings.Contains(path, "..") || strings.HasPrefix(path, "/") {
		return fmt.Errorf("invalid file path: absolute paths and directory traversal not allowed")
	}

	return nil
}

// DefaultDataValidator implements DataValidator
type DefaultDataValidator struct{}

// NewDefaultDataValidator creates a new default data validator
func NewDefaultDataValidator() *DefaultDataValidator {
	return &DefaultDataValidator{}
}

// ValidateTrainingData checks if the training data is valid
func (d *DefaultDataValidator) ValidateTrainingData(inputs [][]float64, targets []float64) error {
	if len(inputs) == 0 {
		return fmt.Errorf("no training data found")
	}

	if len(inputs) != len(targets) {
		return fmt.Errorf("mismatch between inputs (%d) and targets (%d)", len(inputs), len(targets))
	}

	// Check that all input vectors have the same length
	if len(inputs) > 0 {
		expectedLen := len(inputs[0])
		for i, input := range inputs {
			if len(input) != expectedLen {
				return fmt.Errorf("input %d has length %d, expected %d", i, len(input), expectedLen)
			}
		}
	}

	return nil
}

// ValidateInputData validates input data for inference
func (d *DefaultDataValidator) ValidateInputData(inputs []float64) error {
	if len(inputs) == 0 {
		return fmt.Errorf("no input data provided")
	}

	return nil
}

// DefaultDataParser implements DataParser
type DefaultDataParser struct{}

// NewDefaultDataParser creates a new default data parser
func NewDefaultDataParser() *DefaultDataParser {
	return &DefaultDataParser{}
}

// ParseInputString converts comma-separated string to float64 slice
func (d *DefaultDataParser) ParseInputString(data string) ([]float64, error) {
	if strings.TrimSpace(data) == "" {
		return nil, fmt.Errorf("empty input data")
	}

	parts := strings.Split(data, ",")
	inputs := make([]float64, len(parts))

	for i, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid number at position %d: '%s'", i+1, part)
		}
		inputs[i] = value
	}

	return inputs, nil
}
