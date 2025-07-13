// Package datasets implements image dataset loading and preprocessing
// Learning Goal: Understanding image data handling for CNN training and evaluation
package datasets

import (
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
)

// ImageDataset represents a labeled image dataset
// Mathematical Foundation: Image data as tensors with shape [N][H][W][C]
type ImageDataset struct {
	Name     string          // Dataset name (e.g., "MNIST", "CIFAR-10")
	Images   [][][][]float64 // Images: [samples][height][width][channels]
	Labels   []int           // Class labels for each image
	Classes  []string        // Class names
	Width    int             // Image width
	Height   int             // Image height
	Channels int             // Number of channels (1 for grayscale, 3 for RGB)
}

// MNISTDataset represents the MNIST handwritten digit dataset
// Mathematical Foundation: 28x28 grayscale images, 10 classes (0-9)
type MNISTDataset struct {
	TrainImages [][][][]float64 // Training images: [60000][28][28][1]
	TrainLabels []int           // Training labels: [60000]
	TestImages  [][][][]float64 // Test images: [10000][28][28][1]
	TestLabels  []int           // Test labels: [10000]
}

// MNIST file URLs and metadata
const (
	mnistBaseURL     = "http://yann.lecun.com/exdb/mnist/"
	mnistTrainImages = "train-images-idx3-ubyte.gz"
	mnistTrainLabels = "train-labels-idx1-ubyte.gz"
	mnistTestImages  = "t10k-images-idx3-ubyte.gz"
	mnistTestLabels  = "t10k-labels-idx1-ubyte.gz"
)

// LoadMNIST loads the MNIST dataset from files or downloads if needed
// Learning Goal: Understanding binary data format handling and normalization
func LoadMNIST(dataDir string) (*MNISTDataset, error) {
	if dataDir == "" {
		dataDir = "datasets/mnist"
	}

	// Create data directory if it doesn't exist
	err := os.MkdirAll(dataDir, 0755)
	if err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	// Download files if needed
	files := []string{mnistTrainImages, mnistTrainLabels, mnistTestImages, mnistTestLabels}
	for _, filename := range files {
		filePath := filepath.Join(dataDir, filename)
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			fmt.Printf("📥 Downloading %s...\n", filename)
			err = downloadFile(mnistBaseURL+filename, filePath)
			if err != nil {
				return nil, fmt.Errorf("failed to download %s: %w", filename, err)
			}
		}
	}

	// Load training data
	fmt.Printf("📊 Loading MNIST training data...\n")
	trainImages, err := loadMNISTImages(filepath.Join(dataDir, mnistTrainImages))
	if err != nil {
		return nil, fmt.Errorf("failed to load training images: %w", err)
	}

	trainLabels, err := loadMNISTLabels(filepath.Join(dataDir, mnistTrainLabels))
	if err != nil {
		return nil, fmt.Errorf("failed to load training labels: %w", err)
	}

	// Load test data
	fmt.Printf("📊 Loading MNIST test data...\n")
	testImages, err := loadMNISTImages(filepath.Join(dataDir, mnistTestImages))
	if err != nil {
		return nil, fmt.Errorf("failed to load test images: %w", err)
	}

	testLabels, err := loadMNISTLabels(filepath.Join(dataDir, mnistTestLabels))
	if err != nil {
		return nil, fmt.Errorf("failed to load test labels: %w", err)
	}

	fmt.Printf("✅ MNIST loaded: %d train, %d test samples\n", len(trainImages), len(testImages))

	return &MNISTDataset{
		TrainImages: trainImages,
		TrainLabels: trainLabels,
		TestImages:  testImages,
		TestLabels:  testLabels,
	}, nil
}

// loadMNISTImages loads MNIST image data from IDX3 format
// Mathematical Foundation: Pixel normalization to [0,1] range for neural networks
func loadMNISTImages(filename string) ([][][][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("cannot open file: %w", err)
	}
	defer file.Close()

	// Decompress gzip
	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("cannot create gzip reader: %w", err)
	}
	defer gzReader.Close()

	// Read magic number
	var magic uint32
	err = binary.Read(gzReader, binary.BigEndian, &magic)
	if err != nil {
		return nil, fmt.Errorf("cannot read magic number: %w", err)
	}
	if magic != 0x00000803 {
		return nil, fmt.Errorf("invalid magic number for images: %d", magic)
	}

	// Read dimensions
	var numImages, rows, cols uint32
	err = binary.Read(gzReader, binary.BigEndian, &numImages)
	if err != nil {
		return nil, fmt.Errorf("cannot read number of images: %w", err)
	}
	err = binary.Read(gzReader, binary.BigEndian, &rows)
	if err != nil {
		return nil, fmt.Errorf("cannot read number of rows: %w", err)
	}
	err = binary.Read(gzReader, binary.BigEndian, &cols)
	if err != nil {
		return nil, fmt.Errorf("cannot read number of columns: %w", err)
	}

	// Allocate images array
	images := make([][][][]float64, numImages)
	for i := range images {
		images[i] = make([][][]float64, rows)
		for j := range images[i] {
			images[i][j] = make([][]float64, cols)
			for k := range images[i][j] {
				images[i][j][k] = make([]float64, 1) // Grayscale: 1 channel
			}
		}
	}

	// Read image data
	pixelData := make([]byte, rows*cols)
	for i := uint32(0); i < numImages; i++ {
		_, err = io.ReadFull(gzReader, pixelData)
		if err != nil {
			return nil, fmt.Errorf("cannot read image %d: %w", i, err)
		}

		// Convert to normalized float values [0,1]
		for row := uint32(0); row < rows; row++ {
			for col := uint32(0); col < cols; col++ {
				pixelIndex := row*cols + col
				// Normalize pixel value from [0,255] to [0,1]
				images[i][row][col][0] = float64(pixelData[pixelIndex]) / 255.0
			}
		}
	}

	return images, nil
}

// loadMNISTLabels loads MNIST label data from IDX1 format
// Learning Goal: Understanding label encoding for classification tasks
func loadMNISTLabels(filename string) ([]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("cannot open file: %w", err)
	}
	defer file.Close()

	// Decompress gzip
	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("cannot create gzip reader: %w", err)
	}
	defer gzReader.Close()

	// Read magic number
	var magic uint32
	err = binary.Read(gzReader, binary.BigEndian, &magic)
	if err != nil {
		return nil, fmt.Errorf("cannot read magic number: %w", err)
	}
	if magic != 0x00000801 {
		return nil, fmt.Errorf("invalid magic number for labels: %d", magic)
	}

	// Read number of labels
	var numLabels uint32
	err = binary.Read(gzReader, binary.BigEndian, &numLabels)
	if err != nil {
		return nil, fmt.Errorf("cannot read number of labels: %w", err)
	}

	// Read label data
	labelData := make([]byte, numLabels)
	_, err = io.ReadFull(gzReader, labelData)
	if err != nil {
		return nil, fmt.Errorf("cannot read label data: %w", err)
	}

	// Convert to int slice
	labels := make([]int, numLabels)
	for i, label := range labelData {
		labels[i] = int(label)
	}

	return labels, nil
}

// downloadFile downloads a file from URL to local path
func downloadFile(url, filepath string) error {
	// Create the file
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Get the data
	resp, err := http.Get(url) //nolint:gosec // Educational implementation, URL is from constant
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check server response
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Write the body to file
	_, err = io.Copy(out, resp.Body)
	return err
}

// CreateDataset converts MNIST to ImageDataset format
// Learning Goal: Understanding dataset abstraction for unified interface
func (mnist *MNISTDataset) CreateDataset(useTest bool) *ImageDataset {
	var images [][][][]float64
	var labels []int

	if useTest {
		images = mnist.TestImages
		labels = mnist.TestLabels
	} else {
		images = mnist.TrainImages
		labels = mnist.TrainLabels
	}

	classes := []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

	return &ImageDataset{
		Name:     "MNIST",
		Images:   images,
		Labels:   labels,
		Classes:  classes,
		Width:    28,
		Height:   28,
		Channels: 1,
	}
}

// GetBatch returns a batch of images and labels
// Learning Goal: Understanding batch processing for efficient training
func (dataset *ImageDataset) GetBatch(indices []int) ([][][][]float64, []int, error) {
	if len(indices) == 0 {
		return nil, nil, errors.New("empty indices")
	}

	batchImages := make([][][][]float64, len(indices))
	batchLabels := make([]int, len(indices))

	for i, idx := range indices {
		if idx < 0 || idx >= len(dataset.Images) {
			return nil, nil, fmt.Errorf("index %d out of range [0, %d)", idx, len(dataset.Images))
		}
		batchImages[i] = dataset.Images[idx]
		batchLabels[i] = dataset.Labels[idx]
	}

	return batchImages, batchLabels, nil
}

// Shuffle randomly shuffles dataset indices
// Learning Goal: Understanding data shuffling for training optimization
func (dataset *ImageDataset) Shuffle() []int {
	indices := make([]int, len(dataset.Images))
	for i := range indices {
		indices[i] = i
	}

	// Fisher-Yates shuffle
	for i := len(indices) - 1; i > 0; i-- {
		j := rand.Intn(i + 1) //nolint:gosec // Educational implementation, cryptographic randomness not required
		indices[i], indices[j] = indices[j], indices[i]
	}

	return indices
}

// GetImageStatistics calculates basic statistics for the dataset
// Learning Goal: Understanding data distribution analysis
func (dataset *ImageDataset) GetImageStatistics() map[string]float64 {
	if len(dataset.Images) == 0 {
		return map[string]float64{}
	}

	var sum, sumSquared float64
	var min, max = math.Inf(1), math.Inf(-1)
	totalPixels := 0

	// Calculate statistics across all pixels
	for _, image := range dataset.Images {
		for _, row := range image {
			for _, col := range row {
				for _, channel := range col {
					sum += channel
					sumSquared += channel * channel
					if channel < min {
						min = channel
					}
					if channel > max {
						max = channel
					}
					totalPixels++
				}
			}
		}
	}

	mean := sum / float64(totalPixels)
	variance := (sumSquared / float64(totalPixels)) - (mean * mean)
	stddev := math.Sqrt(variance)

	return map[string]float64{
		"mean":    mean,
		"std":     stddev,
		"min":     min,
		"max":     max,
		"samples": float64(len(dataset.Images)),
		"pixels":  float64(totalPixels),
	}
}

// GetClassDistribution returns the distribution of classes in the dataset
// Learning Goal: Understanding class balance analysis
func (dataset *ImageDataset) GetClassDistribution() map[int]int {
	distribution := make(map[int]int)
	for _, label := range dataset.Labels {
		distribution[label]++
	}
	return distribution
}

// NormalizeDataset applies normalization to the entire dataset
// Mathematical Foundation: z = (x - μ) / σ for standardization
func (dataset *ImageDataset) NormalizeDataset(method string) error {
	if len(dataset.Images) == 0 {
		return errors.New("empty dataset")
	}

	switch method {
	case "standard":
		// Z-score normalization: (x - mean) / std
		stats := dataset.GetImageStatistics()
		mean := stats["mean"]
		std := stats["std"]

		if std == 0 {
			return errors.New("zero standard deviation, cannot normalize")
		}

		for i := range dataset.Images {
			for j := range dataset.Images[i] {
				for k := range dataset.Images[i][j] {
					for l := range dataset.Images[i][j][k] {
						dataset.Images[i][j][k][l] = (dataset.Images[i][j][k][l] - mean) / std
					}
				}
			}
		}

	case "minmax":
		// Min-max normalization: (x - min) / (max - min)
		stats := dataset.GetImageStatistics()
		min := stats["min"]
		max := stats["max"]

		if max == min {
			return errors.New("zero range, cannot normalize")
		}

		for i := range dataset.Images {
			for j := range dataset.Images[i] {
				for k := range dataset.Images[i][j] {
					for l := range dataset.Images[i][j][k] {
						dataset.Images[i][j][k][l] = (dataset.Images[i][j][k][l] - min) / (max - min)
					}
				}
			}
		}

	default:
		return fmt.Errorf("unsupported normalization method: %s", method)
	}

	return nil
}

// PrintDatasetInfo displays comprehensive dataset information
// Learning Goal: Understanding dataset analysis and visualization
func (dataset *ImageDataset) PrintDatasetInfo() {
	fmt.Printf("📊 Dataset Information: %s\n", dataset.Name)
	fmt.Printf("   📐 Image size: %dx%dx%d\n", dataset.Height, dataset.Width, dataset.Channels)
	fmt.Printf("   📈 Total samples: %d\n", len(dataset.Images))
	fmt.Printf("   🏷️  Classes: %d (%v)\n", len(dataset.Classes), dataset.Classes)

	// Show class distribution
	distribution := dataset.GetClassDistribution()
	fmt.Printf("   📊 Class distribution:\n")
	for class, count := range distribution {
		percentage := float64(count) / float64(len(dataset.Labels)) * 100
		fmt.Printf("      Class %d: %d samples (%.1f%%)\n", class, count, percentage)
	}

	// Show image statistics
	stats := dataset.GetImageStatistics()
	fmt.Printf("   📊 Pixel statistics:\n")
	fmt.Printf("      Mean: %.4f\n", stats["mean"])
	fmt.Printf("      Std:  %.4f\n", stats["std"])
	fmt.Printf("      Min:  %.4f\n", stats["min"])
	fmt.Printf("      Max:  %.4f\n", stats["max"])
}
