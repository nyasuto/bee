// Package datasets implements comprehensive tests for image dataset functionality
// Learning Goal: Understanding image data validation and processing verification
package datasets

import (
	"reflect"
	"testing"
)

// TestImageDataset tests the basic ImageDataset structure and methods
func TestImageDataset(t *testing.T) {
	t.Run("CreateEmptyDataset", func(t *testing.T) {
		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   nil,
			Labels:   nil,
			Classes:  []string{"0", "1"},
			Width:    28,
			Height:   28,
			Channels: 1,
		}

		if dataset.Name != "TestDataset" {
			t.Errorf("Expected name 'TestDataset', got %s", dataset.Name)
		}
		if dataset.Width != 28 || dataset.Height != 28 || dataset.Channels != 1 {
			t.Errorf("Expected dimensions 28x28x1, got %dx%dx%d", dataset.Height, dataset.Width, dataset.Channels)
		}
	})

	t.Run("CreateDatasetWithSampleData", func(t *testing.T) {
		// Create a small test dataset
		images := make([][][][]float64, 2)
		labels := []int{0, 1}

		// Create 2x2x1 images for testing
		for i := range images {
			images[i] = make([][][]float64, 2)
			for j := range images[i] {
				images[i][j] = make([][]float64, 2)
				for k := range images[i][j] {
					images[i][j][k] = make([]float64, 1)
					images[i][j][k][0] = float64(i + j + k) // Simple test pattern
				}
			}
		}

		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   images,
			Labels:   labels,
			Classes:  []string{"0", "1"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		if len(dataset.Images) != 2 {
			t.Errorf("Expected 2 images, got %d", len(dataset.Images))
		}
		if len(dataset.Labels) != 2 {
			t.Errorf("Expected 2 labels, got %d", len(dataset.Labels))
		}
	})
}

// TestImageDatasetBatch tests batch operations
func TestImageDatasetBatch(t *testing.T) {
	// Create test dataset
	images := make([][][][]float64, 4)
	labels := []int{0, 1, 0, 1}

	for i := range images {
		images[i] = make([][][]float64, 2)
		for j := range images[i] {
			images[i][j] = make([][]float64, 2)
			for k := range images[i][j] {
				images[i][j][k] = make([]float64, 1)
				images[i][j][k][0] = float64(i)
			}
		}
	}

	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   images,
		Labels:   labels,
		Classes:  []string{"0", "1"},
		Width:    2,
		Height:   2,
		Channels: 1,
	}

	t.Run("ValidBatch", func(t *testing.T) {
		indices := []int{0, 2}
		batchImages, batchLabels, err := dataset.GetBatch(indices)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(batchImages) != 2 {
			t.Errorf("Expected batch size 2, got %d", len(batchImages))
		}
		if len(batchLabels) != 2 {
			t.Errorf("Expected batch labels size 2, got %d", len(batchLabels))
		}

		// Verify correct images selected
		if batchImages[0][0][0][0] != 0.0 {
			t.Errorf("Expected first image pixel 0.0, got %f", batchImages[0][0][0][0])
		}
		if batchImages[1][0][0][0] != 2.0 {
			t.Errorf("Expected second image pixel 2.0, got %f", batchImages[1][0][0][0])
		}

		// Verify correct labels
		expectedLabels := []int{0, 0}
		if !reflect.DeepEqual(batchLabels, expectedLabels) {
			t.Errorf("Expected labels %v, got %v", expectedLabels, batchLabels)
		}
	})

	t.Run("EmptyIndices", func(t *testing.T) {
		indices := []int{}
		_, _, err := dataset.GetBatch(indices)
		if err == nil {
			t.Error("Expected error for empty indices")
		}
	})

	t.Run("OutOfRangeIndices", func(t *testing.T) {
		indices := []int{0, 10}
		_, _, err := dataset.GetBatch(indices)
		if err == nil {
			t.Error("Expected error for out of range indices")
		}
	})

	t.Run("NegativeIndices", func(t *testing.T) {
		indices := []int{-1, 0}
		_, _, err := dataset.GetBatch(indices)
		if err == nil {
			t.Error("Expected error for negative indices")
		}
	})
}

// TestImageDatasetShuffle tests the shuffle functionality
func TestImageDatasetShuffle(t *testing.T) {
	// Create test dataset
	images := make([][][][]float64, 10)
	labels := make([]int, 10)

	for i := range images {
		images[i] = make([][][]float64, 1)
		images[i][0] = make([][]float64, 1)
		images[i][0][0] = make([]float64, 1)
		images[i][0][0][0] = float64(i)
		labels[i] = i
	}

	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   images,
		Labels:   labels,
		Classes:  []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
		Width:    1,
		Height:   1,
		Channels: 1,
	}

	t.Run("ShuffleIndices", func(t *testing.T) {
		indices := dataset.Shuffle()

		// Check correct length
		if len(indices) != 10 {
			t.Errorf("Expected 10 indices, got %d", len(indices))
		}

		// Check all indices present
		found := make(map[int]bool)
		for _, idx := range indices {
			if idx < 0 || idx >= 10 {
				t.Errorf("Index out of range: %d", idx)
			}
			found[idx] = true
		}

		if len(found) != 10 {
			t.Errorf("Expected all 10 indices, got %d unique", len(found))
		}
	})

	t.Run("EmptyDatasetShuffle", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "Empty",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		indices := emptyDataset.Shuffle()
		if len(indices) != 0 {
			t.Errorf("Expected empty indices for empty dataset, got %d", len(indices))
		}
	})
}

// TestImageDatasetStatistics tests statistical calculations
func TestImageDatasetStatistics(t *testing.T) {
	// Create test dataset with known values
	images := make([][][][]float64, 2)
	labels := []int{0, 1}

	// First image: all pixels = 0.5
	images[0] = make([][][]float64, 2)
	for j := range images[0] {
		images[0][j] = make([][]float64, 2)
		for k := range images[0][j] {
			images[0][j][k] = make([]float64, 1)
			images[0][j][k][0] = 0.5
		}
	}

	// Second image: all pixels = 1.0
	images[1] = make([][][]float64, 2)
	for j := range images[1] {
		images[1][j] = make([][]float64, 2)
		for k := range images[1][j] {
			images[1][j][k] = make([]float64, 1)
			images[1][j][k][0] = 1.0
		}
	}

	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   images,
		Labels:   labels,
		Classes:  []string{"0", "1"},
		Width:    2,
		Height:   2,
		Channels: 1,
	}

	t.Run("BasicStatistics", func(t *testing.T) {
		stats := dataset.GetImageStatistics()

		// Expected values: 4 pixels with 0.5, 4 pixels with 1.0
		// Mean = (4*0.5 + 4*1.0) / 8 = 0.75
		expectedMean := 0.75
		if abs(stats["mean"]-expectedMean) > 1e-9 {
			t.Errorf("Expected mean %.9f, got %.9f", expectedMean, stats["mean"])
		}

		if stats["min"] != 0.5 {
			t.Errorf("Expected min 0.5, got %f", stats["min"])
		}

		if stats["max"] != 1.0 {
			t.Errorf("Expected max 1.0, got %f", stats["max"])
		}

		if stats["samples"] != 2 {
			t.Errorf("Expected 2 samples, got %f", stats["samples"])
		}

		if stats["pixels"] != 8 {
			t.Errorf("Expected 8 pixels, got %f", stats["pixels"])
		}
	})

	t.Run("EmptyDatasetStatistics", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "Empty",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		stats := emptyDataset.GetImageStatistics()
		if len(stats) != 0 {
			t.Errorf("Expected empty stats for empty dataset, got %v", stats)
		}
	})
}

// TestClassDistribution tests class distribution calculation
func TestClassDistribution(t *testing.T) {
	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   make([][][][]float64, 6), // Placeholder
		Labels:   []int{0, 1, 0, 2, 1, 0},
		Classes:  []string{"0", "1", "2"},
		Width:    1,
		Height:   1,
		Channels: 1,
	}

	t.Run("ValidDistribution", func(t *testing.T) {
		distribution := dataset.GetClassDistribution()

		expected := map[int]int{
			0: 3,
			1: 2,
			2: 1,
		}

		if !reflect.DeepEqual(distribution, expected) {
			t.Errorf("Expected distribution %v, got %v", expected, distribution)
		}
	})

	t.Run("EmptyLabels", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "Empty",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		distribution := emptyDataset.GetClassDistribution()
		if len(distribution) != 0 {
			t.Errorf("Expected empty distribution for empty dataset, got %v", distribution)
		}
	})
}

// TestNormalization tests dataset normalization methods
func TestNormalization(t *testing.T) {
	// Create test dataset with known values for normalization testing
	images := make([][][][]float64, 2)
	labels := []int{0, 1}

	// Image with values [0, 100, 200, 255] (simulating pixel values)
	images[0] = make([][][]float64, 2)
	images[0][0] = make([][]float64, 2)
	images[0][0][0] = []float64{0}
	images[0][0][1] = []float64{100}
	images[0][1] = make([][]float64, 2)
	images[0][1][0] = []float64{200}
	images[0][1][1] = []float64{255}

	// Second image with same pattern
	images[1] = make([][][]float64, 2)
	images[1][0] = make([][]float64, 2)
	images[1][0][0] = []float64{0}
	images[1][0][1] = []float64{100}
	images[1][1] = make([][]float64, 2)
	images[1][1][0] = []float64{200}
	images[1][1][1] = []float64{255}

	t.Run("MinMaxNormalization", func(t *testing.T) {
		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   copyImages(images), // Use copy to avoid modifying original
			Labels:   labels,
			Classes:  []string{"0", "1"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		err := dataset.NormalizeDataset("minmax")
		if err != nil {
			t.Fatalf("Normalization failed: %v", err)
		}

		// After min-max normalization: min=0 -> 0, max=255 -> 1
		if dataset.Images[0][0][0][0] != 0.0 {
			t.Errorf("Expected normalized min 0.0, got %f", dataset.Images[0][0][0][0])
		}
		if abs(dataset.Images[0][1][1][0]-1.0) > 1e-9 {
			t.Errorf("Expected normalized max 1.0, got %f", dataset.Images[0][1][1][0])
		}
	})

	t.Run("StandardNormalization", func(t *testing.T) {
		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   copyImages(images), // Use copy to avoid modifying original
			Labels:   labels,
			Classes:  []string{"0", "1"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		err := dataset.NormalizeDataset("standard")
		if err != nil {
			t.Fatalf("Normalization failed: %v", err)
		}

		// Verify that mean is approximately 0 after standardization
		stats := dataset.GetImageStatistics()
		if abs(stats["mean"]) > 1e-9 {
			t.Errorf("Expected mean ~0 after standardization, got %f", stats["mean"])
		}
	})

	t.Run("UnsupportedNormalization", func(t *testing.T) {
		dataset := &ImageDataset{
			Name:     "TestDataset",
			Images:   copyImages(images),
			Labels:   labels,
			Classes:  []string{"0", "1"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		err := dataset.NormalizeDataset("unsupported")
		if err == nil {
			t.Error("Expected error for unsupported normalization method")
		}
	})

	t.Run("EmptyDatasetNormalization", func(t *testing.T) {
		emptyDataset := &ImageDataset{
			Name:     "Empty",
			Images:   [][][][]float64{},
			Labels:   []int{},
			Classes:  []string{},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		err := emptyDataset.NormalizeDataset("standard")
		if err == nil {
			t.Error("Expected error for normalizing empty dataset")
		}
	})
}

// TestMNISTDataset tests MNIST-specific functionality
func TestMNISTDataset(t *testing.T) {
	t.Run("MNISTDatasetCreation", func(t *testing.T) {
		// Create mock MNIST data
		trainImages := make([][][][]float64, 2)
		trainLabels := []int{0, 1}
		testImages := make([][][][]float64, 1)
		testLabels := []int{2}

		// Initialize with 28x28x1 structure
		for i := range trainImages {
			trainImages[i] = make([][][]float64, 28)
			for j := range trainImages[i] {
				trainImages[i][j] = make([][]float64, 28)
				for k := range trainImages[i][j] {
					trainImages[i][j][k] = make([]float64, 1)
					trainImages[i][j][k][0] = float64(i) // Simple test pattern
				}
			}
		}

		for i := range testImages {
			testImages[i] = make([][][]float64, 28)
			for j := range testImages[i] {
				testImages[i][j] = make([][]float64, 28)
				for k := range testImages[i][j] {
					testImages[i][j][k] = make([]float64, 1)
					testImages[i][j][k][0] = float64(i + 10) // Different pattern
				}
			}
		}

		mnist := &MNISTDataset{
			TrainImages: trainImages,
			TrainLabels: trainLabels,
			TestImages:  testImages,
			TestLabels:  testLabels,
		}

		// Test training dataset conversion
		trainDataset := mnist.CreateDataset(false)
		if trainDataset.Name != "MNIST" {
			t.Errorf("Expected name 'MNIST', got %s", trainDataset.Name)
		}
		if len(trainDataset.Images) != 2 {
			t.Errorf("Expected 2 training images, got %d", len(trainDataset.Images))
		}
		if trainDataset.Width != 28 || trainDataset.Height != 28 || trainDataset.Channels != 1 {
			t.Errorf("Expected 28x28x1, got %dx%dx%d", trainDataset.Height, trainDataset.Width, trainDataset.Channels)
		}

		// Test test dataset conversion
		testDataset := mnist.CreateDataset(true)
		if len(testDataset.Images) != 1 {
			t.Errorf("Expected 1 test image, got %d", len(testDataset.Images))
		}
	})
}

// TestPrintDatasetInfo tests the dataset info printing (basic functionality)
func TestPrintDatasetInfo(t *testing.T) {
	dataset := &ImageDataset{
		Name:     "TestDataset",
		Images:   make([][][][]float64, 5),
		Labels:   []int{0, 1, 0, 1, 2},
		Classes:  []string{"0", "1", "2"},
		Width:    28,
		Height:   28,
		Channels: 1,
	}

	// Initialize images with placeholder data
	for i := range dataset.Images {
		dataset.Images[i] = make([][][]float64, 28)
		for j := range dataset.Images[i] {
			dataset.Images[i][j] = make([][]float64, 28)
			for k := range dataset.Images[i][j] {
				dataset.Images[i][j][k] = make([]float64, 1)
				dataset.Images[i][j][k][0] = 0.5 // Constant value for simplicity
			}
		}
	}

	t.Run("PrintInfo", func(t *testing.T) {
		// This test just ensures the function doesn't panic
		// In a real scenario, you might capture stdout to verify output
		dataset.PrintDatasetInfo()
	})
}

// Helper functions for testing

// copyImages creates a deep copy of images array for testing
func copyImages(original [][][][]float64) [][][][]float64 {
	copy := make([][][][]float64, len(original))
	for i := range original {
		copy[i] = make([][][]float64, len(original[i]))
		for j := range original[i] {
			copy[i][j] = make([][]float64, len(original[i][j]))
			for k := range original[i][j] {
				copy[i][j][k] = make([]float64, len(original[i][j][k]))
				for l := range original[i][j][k] {
					copy[i][j][k][l] = original[i][j][k][l]
				}
			}
		}
	}
	return copy
}

// abs returns the absolute value of a float64
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// TestEdgeCases tests various edge cases and error conditions
func TestEdgeCases(t *testing.T) {
	t.Run("LoadMNISTInvalidDirectory", func(t *testing.T) {
		// Test with non-existent directory that we can't create (permission test)
		_, err := LoadMNIST("/invalid/read/only/path")
		if err == nil {
			t.Error("Expected error for invalid directory path")
		}
	})

	t.Run("GetBatchSingleIndex", func(t *testing.T) {
		// Create minimal dataset
		images := make([][][][]float64, 1)
		images[0] = make([][][]float64, 1)
		images[0][0] = make([][]float64, 1)
		images[0][0][0] = []float64{0.5}

		dataset := &ImageDataset{
			Name:     "SingleImage",
			Images:   images,
			Labels:   []int{0},
			Classes:  []string{"0"},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		batchImages, batchLabels, err := dataset.GetBatch([]int{0})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if len(batchImages) != 1 || len(batchLabels) != 1 {
			t.Errorf("Expected batch size 1, got images: %d, labels: %d", len(batchImages), len(batchLabels))
		}
	})

	t.Run("NormalizationZeroVariance", func(t *testing.T) {
		// Create dataset with constant pixel values
		images := make([][][][]float64, 2)
		for i := range images {
			images[i] = make([][][]float64, 2)
			for j := range images[i] {
				images[i][j] = make([][]float64, 2)
				for k := range images[i][j] {
					images[i][j][k] = []float64{0.5} // All pixels same value
				}
			}
		}

		dataset := &ImageDataset{
			Name:     "ConstantDataset",
			Images:   images,
			Labels:   []int{0, 1},
			Classes:  []string{"0", "1"},
			Width:    2,
			Height:   2,
			Channels: 1,
		}

		err := dataset.NormalizeDataset("standard")
		if err == nil {
			t.Error("Expected error for zero variance in standard normalization")
		}
	})

	t.Run("NormalizationZeroRange", func(t *testing.T) {
		// Create dataset with constant pixel values for min-max test
		images := make([][][][]float64, 1)
		images[0] = make([][][]float64, 1)
		images[0][0] = make([][]float64, 1)
		images[0][0][0] = []float64{0.5}

		dataset := &ImageDataset{
			Name:     "ConstantDataset",
			Images:   images,
			Labels:   []int{0},
			Classes:  []string{"0"},
			Width:    1,
			Height:   1,
			Channels: 1,
		}

		err := dataset.NormalizeDataset("minmax")
		if err == nil {
			t.Error("Expected error for zero range in minmax normalization")
		}
	})
}

// TestMNISTIntegration tests MNIST loading integration (requires internet)
func TestMNISTIntegration(t *testing.T) {
	// This test is marked as integration and may be skipped in CI
	if testing.Short() {
		t.Skip("Skipping MNIST integration test in short mode")
	}

	t.Run("LoadMNISTWithTempDir", func(t *testing.T) {
		// Create temporary directory for test
		tempDir := t.TempDir()

		// Note: This test will actually download MNIST data if not cached
		// In a real scenario, you might want to use mock data or check if internet is available
		mnist, err := LoadMNIST(tempDir)
		if err != nil {
			t.Skipf("Skipping MNIST test due to download failure: %v", err)
			return
		}

		// Verify dataset structure
		if len(mnist.TrainImages) == 0 {
			t.Error("Expected non-empty training images")
		}
		if len(mnist.TrainLabels) == 0 {
			t.Error("Expected non-empty training labels")
		}
		if len(mnist.TestImages) == 0 {
			t.Error("Expected non-empty test images")
		}
		if len(mnist.TestLabels) == 0 {
			t.Error("Expected non-empty test labels")
		}

		// Verify image dimensions
		if len(mnist.TrainImages) > 0 {
			if len(mnist.TrainImages[0]) != 28 {
				t.Errorf("Expected height 28, got %d", len(mnist.TrainImages[0]))
			}
			if len(mnist.TrainImages[0][0]) != 28 {
				t.Errorf("Expected width 28, got %d", len(mnist.TrainImages[0][0]))
			}
			if len(mnist.TrainImages[0][0][0]) != 1 {
				t.Errorf("Expected 1 channel, got %d", len(mnist.TrainImages[0][0][0]))
			}
		}

		// Test conversion to dataset format
		trainDataset := mnist.CreateDataset(false)
		testDataset := mnist.CreateDataset(true)

		trainDataset.PrintDatasetInfo()
		testDataset.PrintDatasetInfo()
	})
}
