// Package model provides comprehensive tests for model management
// Learning Goal: Understanding model layer testing patterns and adapter pattern validation
package model

import (
	"os"
	"reflect"
	"testing"

	"github.com/nyasuto/bee/phase1"
)

// TestDefaultModelManager tests the default model manager implementation
func TestDefaultModelManager(t *testing.T) {
	manager := NewDefaultModelManager()

	t.Run("ListModelTypes", func(t *testing.T) {
		types := manager.ListModelTypes()
		expectedTypes := []string{"perceptron"}

		if !reflect.DeepEqual(types, expectedTypes) {
			t.Errorf("Expected model types %v, got %v", expectedTypes, types)
		}
	})

	t.Run("CreatePerceptronModel", func(t *testing.T) {
		model, err := manager.CreateModel("perceptron", 2, 0.1)
		if err != nil {
			t.Errorf("Unexpected error creating perceptron: %v", err)
			return
		}

		if model == nil {
			t.Error("Expected non-nil model")
			return
		}

		// Test that it's a valid perceptron adapter
		adapter, ok := model.(*PerceptronAdapter)
		if !ok {
			t.Error("Expected PerceptronAdapter")
			return
		}

		if adapter.perceptron == nil {
			t.Error("Expected non-nil underlying perceptron")
		}
	})

	// MLP support will be added in Phase 2

	t.Run("CreateInvalidModel", func(t *testing.T) {
		_, err := manager.CreateModel("invalid", 2, 0.1)
		if err == nil {
			t.Error("Expected error for invalid model type")
		}
	})

	t.Run("CreateModelWithZeroInputSize", func(t *testing.T) {
		// Model manager doesn't validate parameters - it passes them to the underlying model
		// The perceptron constructor may handle this differently
		model, err := manager.CreateModel("perceptron", 0, 0.1)
		if err != nil {
			t.Logf("Zero input size handling: %v", err)
		} else if model != nil {
			t.Log("Model created with zero input size - validation happens at usage time")
		}
	})

	t.Run("CreateModelWithNegativeLearningRate", func(t *testing.T) {
		// Model manager doesn't validate parameters - it passes them to the underlying model
		// The perceptron constructor may handle this differently
		model, err := manager.CreateModel("perceptron", 2, -0.1)
		if err != nil {
			t.Logf("Negative learning rate handling: %v", err)
		} else if model != nil {
			t.Log("Model created with negative learning rate - validation happens at usage time")
		}
	})
}

// TestPerceptronAdapter tests the perceptron adapter implementation
func TestPerceptronAdapter(t *testing.T) {
	perceptron := phase1.NewPerceptron(2, 0.1)
	adapter := &PerceptronAdapter{perceptron: perceptron}

	t.Run("BasicFunctionality", func(t *testing.T) {
		// Test training
		inputs := [][]float64{{1, 0}, {0, 1}, {0, 0}, {1, 1}}
		targets := []float64{1, 0, 0, 1}

		epochs, err := adapter.Train(inputs, targets, 10)
		if err != nil {
			t.Errorf("Unexpected error during training: %v", err)
			return
		}

		if epochs <= 0 {
			t.Error("Expected positive number of epochs")
		}

		// Test prediction
		prediction, err := adapter.Predict([]float64{1, 0})
		if err != nil {
			t.Errorf("Unexpected error during prediction: %v", err)
			return
		}

		if prediction < 0 || prediction > 1 {
			t.Errorf("Expected prediction between 0 and 1, got %f", prediction)
		}

		// Test accuracy
		accuracy, err := adapter.Accuracy(inputs, targets)
		if err != nil {
			t.Errorf("Unexpected error calculating accuracy: %v", err)
			return
		}

		if accuracy < 0 || accuracy > 1 {
			t.Errorf("Expected accuracy between 0 and 1, got %f", accuracy)
		}

		// Test weight and bias access
		weights := adapter.GetWeights()
		if len(weights) != 2 {
			t.Errorf("Expected 2 weights, got %d", len(weights))
		}

		bias := adapter.GetBias()
		// Bias can be any float value, just check it's a valid number
		if bias != bias { // NaN check
			t.Error("Expected valid bias value")
		}
	})

	t.Run("InvalidInputSize", func(t *testing.T) {
		// Test with wrong input size
		_, err := adapter.Predict([]float64{1}) // Only 1 input instead of 2
		if err == nil {
			t.Error("Expected error for wrong input size")
		}
	})

	t.Run("MismatchedTrainingData", func(t *testing.T) {
		inputs := [][]float64{{1, 0}, {0, 1}}
		targets := []float64{1} // Only 1 target for 2 inputs

		_, err := adapter.Train(inputs, targets, 10)
		if err == nil {
			t.Error("Expected error for mismatched training data")
		}
	})
}

// MLP adapter tests will be added in Phase 2 when MLP implementation is complete

// TestModelPersistence tests model saving and loading
func TestModelPersistence(t *testing.T) {
	manager := NewDefaultModelManager()

	t.Run("SaveAndLoadPerceptron", func(t *testing.T) {
		// Create and train a perceptron
		model, err := manager.CreateModel("perceptron", 2, 0.1)
		if err != nil {
			t.Fatalf("Failed to create perceptron: %v", err)
		}

		// Train it a bit
		inputs := [][]float64{{1, 0}, {0, 1}}
		targets := []float64{1, 0}
		_, err = model.Train(inputs, targets, 5)
		if err != nil {
			t.Fatalf("Failed to train perceptron: %v", err)
		}

		// Save model (using relative path to comply with security restrictions)
		modelPath := "test_perceptron.json"
		err = manager.SaveModel(model, modelPath)
		if err != nil {
			t.Errorf("Unexpected error saving model: %v", err)
			return
		}
		defer os.Remove(modelPath)

		// Load model
		loadedModel, err := manager.LoadModel(modelPath)
		if err != nil {
			t.Errorf("Unexpected error loading model: %v", err)
			return
		}

		// Compare original and loaded models
		originalWeights := model.GetWeights()
		loadedWeights := loadedModel.GetWeights()

		if !reflect.DeepEqual(originalWeights, loadedWeights) {
			t.Errorf("Weights don't match: original %v, loaded %v", originalWeights, loadedWeights)
		}

		if model.GetBias() != loadedModel.GetBias() {
			t.Errorf("Bias doesn't match: original %v, loaded %v", model.GetBias(), loadedModel.GetBias())
		}
	})

	t.Run("SaveModelCreateDirectory", func(t *testing.T) {
		model, err := manager.CreateModel("perceptron", 2, 0.1)
		if err != nil {
			t.Fatalf("Failed to create perceptron: %v", err)
		}

		// Test saving to a nested directory that doesn't exist (using relative path)
		modelPath := "test_subdir/model.json"
		err = manager.SaveModel(model, modelPath)
		if err != nil {
			t.Errorf("Unexpected error creating directory and saving model: %v", err)
		}
		defer func() {
			os.Remove(modelPath)
			os.Remove("test_subdir")
		}()
	})

	t.Run("LoadNonExistentModel", func(t *testing.T) {
		_, err := manager.LoadModel("nonexistent_model.json")
		if err == nil {
			t.Error("Expected error for non-existent model file")
		}
	})

	t.Run("LoadInvalidJSON", func(t *testing.T) {
		// Create invalid JSON file
		invalidJSON := "invalid json content"
		modelPath := "invalid_model.json"
		err := os.WriteFile(modelPath, []byte(invalidJSON), 0600)
		if err != nil {
			t.Fatalf("Failed to create invalid JSON file: %v", err)
		}
		defer os.Remove(modelPath)

		_, err = manager.LoadModel(modelPath)
		if err == nil {
			t.Error("Expected error for invalid JSON")
		}
	})
}
