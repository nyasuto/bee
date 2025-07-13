// Package datasets implements comprehensive tests for MNIST data loading functionality
// Learning Goal: Understanding MNIST dataset structure and binary file parsing
package datasets

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

// TestLoadMNISTImages tests the MNIST image loading functionality
func TestLoadMNISTImages(t *testing.T) {
	t.Run("ValidMNISTImagesFile", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create mock MNIST images file
		imagesFile := filepath.Join(tempDir, "t10k-images.idx3-ubyte")

		// MNIST image file format:
		// - Magic number: 2051 (0x00000803)
		// - Number of images: 2
		// - Number of rows: 28
		// - Number of columns: 28
		// - Image data: 28x28 pixels per image
		var buf bytes.Buffer

		// Write header
		binary.Write(&buf, binary.BigEndian, uint32(2051)) // Magic number
		binary.Write(&buf, binary.BigEndian, uint32(2))    // Number of images
		binary.Write(&buf, binary.BigEndian, uint32(28))   // Rows
		binary.Write(&buf, binary.BigEndian, uint32(28))   // Columns

		// Write image data (2 images of 28x28 = 1568 bytes each)
		imageData := make([]byte, 28*28)
		for i := 0; i < len(imageData); i++ {
			imageData[i] = byte(i % 256) // Pattern data
		}

		// Write first image
		buf.Write(imageData)

		// Write second image (different pattern)
		for i := 0; i < len(imageData); i++ {
			imageData[i] = byte((i + 128) % 256) // Different pattern
		}
		buf.Write(imageData)

		// Compress and save to file
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(imagesFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write mock images file: %v", err)
		}

		// Test loading
		images, err := loadMNISTImages(imagesFile)
		if err != nil {
			t.Fatalf("Failed to load MNIST images: %v", err)
		}

		// Verify results
		if len(images) != 2 {
			t.Errorf("Expected 2 images, got %d", len(images))
		}

		// Check image dimensions
		for i, image := range images {
			if len(image) != 28 {
				t.Errorf("Image %d: expected 28 rows, got %d", i, len(image))
			}
			for j, row := range image {
				if len(row) != 28 {
					t.Errorf("Image %d, row %d: expected 28 columns, got %d", i, j, len(row))
				}
				for k, pixel := range row {
					if len(pixel) != 1 {
						t.Errorf("Image %d, pixel [%d][%d]: expected 1 channel, got %d", i, j, k, len(pixel))
					}
				}
			}
		}

		// Verify pixel data patterns
		firstPixel := images[0][0][0][0]
		if firstPixel != 0.0 {
			t.Errorf("Expected first pixel to be 0.0, got %f", firstPixel)
		}

		// Check that images are different
		if images[0][0][1][0] == images[1][0][1][0] {
			t.Error("Expected different pixel patterns between images")
		}
	})

	t.Run("NonExistentFile", func(t *testing.T) {
		_, err := loadMNISTImages("/non/existent/file.idx3-ubyte")
		if err == nil {
			t.Error("Expected error for non-existent file")
		}
	})

	t.Run("InvalidMagicNumber", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create file with invalid magic number
		imagesFile := filepath.Join(tempDir, "invalid-images.idx3-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(1234)) // Invalid magic number
		binary.Write(&buf, binary.BigEndian, uint32(1))
		binary.Write(&buf, binary.BigEndian, uint32(28))
		binary.Write(&buf, binary.BigEndian, uint32(28))

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(imagesFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write invalid file: %v", err)
		}

		_, err = loadMNISTImages(imagesFile)
		if err == nil {
			t.Error("Expected error for invalid magic number")
		}
	})

	t.Run("TruncatedFile", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create truncated file (header only)
		imagesFile := filepath.Join(tempDir, "truncated-images.idx3-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(2051))
		binary.Write(&buf, binary.BigEndian, uint32(1))
		binary.Write(&buf, binary.BigEndian, uint32(28))
		binary.Write(&buf, binary.BigEndian, uint32(28))
		// No image data

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(imagesFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write truncated file: %v", err)
		}

		_, err = loadMNISTImages(imagesFile)
		if err == nil {
			t.Error("Expected error for truncated file")
		}
	})

	t.Run("ZeroImages", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create file with 0 images
		imagesFile := filepath.Join(tempDir, "zero-images.idx3-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(2051))
		binary.Write(&buf, binary.BigEndian, uint32(0)) // 0 images
		binary.Write(&buf, binary.BigEndian, uint32(28))
		binary.Write(&buf, binary.BigEndian, uint32(28))

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(imagesFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write zero images file: %v", err)
		}

		images, err := loadMNISTImages(imagesFile)
		if err != nil {
			t.Fatalf("Unexpected error for zero images: %v", err)
		}

		if len(images) != 0 {
			t.Errorf("Expected 0 images, got %d", len(images))
		}
	})
}

// TestLoadMNISTLabels tests the MNIST label loading functionality
func TestLoadMNISTLabels(t *testing.T) {
	t.Run("ValidMNISTLabelsFile", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create mock MNIST labels file
		labelsFile := filepath.Join(tempDir, "t10k-labels.idx1-ubyte")

		// MNIST label file format:
		// - Magic number: 2049 (0x00000801)
		// - Number of labels: 5
		// - Label data: 1 byte per label
		var buf bytes.Buffer

		// Write header
		binary.Write(&buf, binary.BigEndian, uint32(2049)) // Magic number
		binary.Write(&buf, binary.BigEndian, uint32(5))    // Number of labels

		// Write label data
		labels := []byte{0, 1, 2, 3, 4} // Labels 0-4
		buf.Write(labels)

		// Compress and save to file
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(labelsFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write mock labels file: %v", err)
		}

		// Test loading
		loadedLabels, err := loadMNISTLabels(labelsFile)
		if err != nil {
			t.Fatalf("Failed to load MNIST labels: %v", err)
		}

		// Verify results
		if len(loadedLabels) != 5 {
			t.Errorf("Expected 5 labels, got %d", len(loadedLabels))
		}

		// Check label values
		expectedLabels := []int{0, 1, 2, 3, 4}
		for i, expected := range expectedLabels {
			if loadedLabels[i] != expected {
				t.Errorf("Label %d: expected %d, got %d", i, expected, loadedLabels[i])
			}
		}
	})

	t.Run("NonExistentFile", func(t *testing.T) {
		_, err := loadMNISTLabels("/non/existent/file.idx1-ubyte")
		if err == nil {
			t.Error("Expected error for non-existent file")
		}
	})

	t.Run("InvalidMagicNumber", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create file with invalid magic number
		labelsFile := filepath.Join(tempDir, "invalid-labels.idx1-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(5678)) // Invalid magic number
		binary.Write(&buf, binary.BigEndian, uint32(1))

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(labelsFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write invalid file: %v", err)
		}

		_, err = loadMNISTLabels(labelsFile)
		if err == nil {
			t.Error("Expected error for invalid magic number")
		}
	})

	t.Run("TruncatedFile", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create truncated file (header only)
		labelsFile := filepath.Join(tempDir, "truncated-labels.idx1-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(2049))
		binary.Write(&buf, binary.BigEndian, uint32(3)) // Claims 3 labels
		// No label data

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(labelsFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write truncated file: %v", err)
		}

		_, err = loadMNISTLabels(labelsFile)
		if err == nil {
			t.Error("Expected error for truncated file")
		}
	})

	t.Run("ZeroLabels", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create file with 0 labels
		labelsFile := filepath.Join(tempDir, "zero-labels.idx1-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(2049))
		binary.Write(&buf, binary.BigEndian, uint32(0)) // 0 labels

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(labelsFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write zero labels file: %v", err)
		}

		labels, err := loadMNISTLabels(labelsFile)
		if err != nil {
			t.Fatalf("Unexpected error for zero labels: %v", err)
		}

		if len(labels) != 0 {
			t.Errorf("Expected 0 labels, got %d", len(labels))
		}
	})

	t.Run("VariousLabelValues", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create file with all possible MNIST labels (0-9)
		labelsFile := filepath.Join(tempDir, "all-labels.idx1-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(2049))
		binary.Write(&buf, binary.BigEndian, uint32(10)) // 10 labels

		// Write all digits 0-9
		allLabels := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
		buf.Write(allLabels)

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(labelsFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write all labels file: %v", err)
		}

		labels, err := loadMNISTLabels(labelsFile)
		if err != nil {
			t.Fatalf("Failed to load all labels: %v", err)
		}

		// Verify all labels
		if len(labels) != 10 {
			t.Errorf("Expected 10 labels, got %d", len(labels))
		}

		for i := 0; i < 10; i++ {
			if labels[i] != i {
				t.Errorf("Label %d: expected %d, got %d", i, i, labels[i])
			}
		}
	})
}

// TestMNISTFileHeaderParsing tests edge cases in header parsing
func TestMNISTFileHeaderParsing(t *testing.T) {
	t.Run("PartialHeader", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create file with partial header (only magic number)
		partialFile := filepath.Join(tempDir, "partial-header.idx3-ubyte")
		var buf bytes.Buffer

		binary.Write(&buf, binary.BigEndian, uint32(2051)) // Only magic number

		// Compress data
		var gzipBuf bytes.Buffer
		gzipWriter := gzip.NewWriter(&gzipBuf)
		_, err = gzipWriter.Write(buf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write to gzip: %v", err)
		}
		err = gzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close gzip writer: %v", err)
		}

		err = ioutil.WriteFile(partialFile, gzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write partial file: %v", err)
		}

		_, err = loadMNISTImages(partialFile)
		if err == nil {
			t.Error("Expected error for partial header")
		}
	})

	t.Run("EmptyFile", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create empty file
		emptyFile := filepath.Join(tempDir, "empty.idx3-ubyte")
		err = ioutil.WriteFile(emptyFile, []byte{}, 0644)
		if err != nil {
			t.Fatalf("Failed to write empty file: %v", err)
		}

		_, err = loadMNISTImages(emptyFile)
		if err == nil {
			t.Error("Expected error for empty file")
		}
	})
}

// TestMNISTDataConsistency tests consistency between images and labels
func TestMNISTDataConsistency(t *testing.T) {
	t.Run("MatchingCounts", func(t *testing.T) {
		// Create temporary directory
		tempDir, err := ioutil.TempDir("", "mnist_test")
		if err != nil {
			t.Fatalf("Failed to create temp dir: %v", err)
		}
		defer os.RemoveAll(tempDir)

		// Create matching images and labels files
		numSamples := 3

		// Create images file
		imagesFile := filepath.Join(tempDir, "test-images.idx3-ubyte")
		var imgBuf bytes.Buffer
		binary.Write(&imgBuf, binary.BigEndian, uint32(2051))
		binary.Write(&imgBuf, binary.BigEndian, uint32(numSamples))
		binary.Write(&imgBuf, binary.BigEndian, uint32(28))
		binary.Write(&imgBuf, binary.BigEndian, uint32(28))

		// Add image data
		imageData := make([]byte, 28*28)
		for i := 0; i < numSamples; i++ {
			for j := 0; j < len(imageData); j++ {
				imageData[j] = byte((i*50 + j) % 256)
			}
			imgBuf.Write(imageData)
		}

		// Compress images data
		var imgGzipBuf bytes.Buffer
		imgGzipWriter := gzip.NewWriter(&imgGzipBuf)
		_, err = imgGzipWriter.Write(imgBuf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write images to gzip: %v", err)
		}
		err = imgGzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close images gzip writer: %v", err)
		}

		err = ioutil.WriteFile(imagesFile, imgGzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write images file: %v", err)
		}

		// Create labels file
		labelsFile := filepath.Join(tempDir, "test-labels.idx1-ubyte")
		var lblBuf bytes.Buffer
		binary.Write(&lblBuf, binary.BigEndian, uint32(2049))
		binary.Write(&lblBuf, binary.BigEndian, uint32(numSamples))

		labels := []byte{7, 2, 9} // Sample labels
		lblBuf.Write(labels)

		// Compress labels data
		var lblGzipBuf bytes.Buffer
		lblGzipWriter := gzip.NewWriter(&lblGzipBuf)
		_, err = lblGzipWriter.Write(lblBuf.Bytes())
		if err != nil {
			t.Fatalf("Failed to write labels to gzip: %v", err)
		}
		err = lblGzipWriter.Close()
		if err != nil {
			t.Fatalf("Failed to close labels gzip writer: %v", err)
		}

		err = ioutil.WriteFile(labelsFile, lblGzipBuf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("Failed to write labels file: %v", err)
		}

		// Load both files
		images, err := loadMNISTImages(imagesFile)
		if err != nil {
			t.Fatalf("Failed to load images: %v", err)
		}

		loadedLabels, err := loadMNISTLabels(labelsFile)
		if err != nil {
			t.Fatalf("Failed to load labels: %v", err)
		}

		// Verify consistency
		if len(images) != len(loadedLabels) {
			t.Errorf("Mismatch between images (%d) and labels (%d)", len(images), len(loadedLabels))
		}

		if len(images) != numSamples {
			t.Errorf("Expected %d images, got %d", numSamples, len(images))
		}

		if len(loadedLabels) != numSamples {
			t.Errorf("Expected %d labels, got %d", numSamples, len(loadedLabels))
		}

		// Verify specific label values
		expectedLabels := []int{7, 2, 9}
		for i, expected := range expectedLabels {
			if loadedLabels[i] != expected {
				t.Errorf("Label %d: expected %d, got %d", i, expected, loadedLabels[i])
			}
		}
	})
}
